// Offline compatibility probe, not an extension backend or video-quality claim.
// Runs the official FSR3 provider with constant depth, zero MV and reset history.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <ffx_api_loader.h>
#include <dx12/ffx_api_dx12.h>
#include <ffx_upscale.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char* operation) {
    if (FAILED(hr)) throw std::runtime_error(std::string(operation)+" HRESULT="+std::to_string(uint32_t(hr)));
}
static void apiCheck(uint32_t code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation)+" FFX code="+std::to_string(code));
}
static std::string escape(const std::string& value) {
    std::string result; for (unsigned char c:value) { if(c=='"'||c=='\\')result+='\\'; if(c>=32)result+=c; }
    return result;
}
static std::string utf8(const wchar_t* value) {
    int n=WideCharToMultiByte(CP_UTF8,0,value,-1,nullptr,0,nullptr,nullptr);
    std::string result(n,'\0'); WideCharToMultiByte(CP_UTF8,0,value,-1,result.data(),n,nullptr,nullptr); result.pop_back(); return result;
}
static int ffxErrors=0;
static void message(uint32_t type, const wchar_t* value) {
    if(type==FFX_API_MESSAGE_TYPE_ERROR)ffxErrors++;
    std::cerr<<"FFX "<<type<<": "<<utf8(value)<<"\n";
}
struct Gpu {
    ComPtr<ID3D12Device> device; ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence; HANDLE event=nullptr; uint64_t serial=0;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    std::string name;
    Gpu() {
        ComPtr<IDXGIFactory6> factory; check(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)),"DXGI factory");
        for(UINT i=0;;i++) {
            ComPtr<IDXGIAdapter1> adapter;
            if(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
            DXGI_ADAPTER_DESC1 desc{}; check(adapter->GetDesc1(&desc),"adapter description");
            if(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)continue;
            if(SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)))) { name=utf8(desc.Description); break; }
        }
        if(!device)throw std::runtime_error("No hardware D3D12 feature-level-12 adapter; WARP prohibited");
        D3D12_COMMAND_QUEUE_DESC desc{}; desc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)),"command queue");
        check(device->CreateCommandAllocator(desc.Type,IID_PPV_ARGS(&allocator)),"allocator");
        check(device->CreateCommandList(0,desc.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&commands)),"command list");
        check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
        event=CreateEvent(nullptr,FALSE,FALSE,nullptr); if(!event)throw std::runtime_error("Fence event creation failed");
    }
    ~Gpu() { if(event)CloseHandle(event); }
    void wait() {
        check(queue->Signal(fence.Get(),++serial),"queue signal");
        if(fence->GetCompletedValue()<serial) {
            check(fence->SetEventOnCompletion(serial,event),"fence completion");
            if(WaitForSingleObject(event,10000)!=WAIT_OBJECT_0)throw std::runtime_error("GPU completion timeout; stop this process");
        }
        check(device->GetDeviceRemovedReason(),"device removed reason");
    }
    void barrier(ID3D12Resource* resource,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,from,to}; commands->ResourceBarrier(1,&b);
    }
    ComPtr<ID3D12Resource> buffer(uint64_t bytes,D3D12_HEAP_TYPE heap,D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES h{};h.Type=heap; ComPtr<ID3D12Resource> r;
        check(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&r)),"buffer"); return r;
    }
    ComPtr<ID3D12Resource> texture(UINT w,UINT h,DXGI_FORMAT format,const void* data,UINT bytesPerPixel) {
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=w;d.Height=h;d.DepthOrArraySize=1;d.MipLevels=1;d.Format=format;d.SampleDesc.Count=1;
        if(!data)d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT; ComPtr<ID3D12Resource> r;
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,data?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&r)),"texture");
        if(data) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 size; device->GetCopyableFootprints(&d,0,1,0,&fp,nullptr,nullptr,&size);
            auto upload=buffer(size,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped=nullptr;D3D12_RANGE empty{0,0};check(upload->Map(0,&empty,&mapped),"upload map");
            for(UINT y=0;y<h;y++)memcpy(static_cast<char*>(mapped)+fp.Offset+y*fp.Footprint.RowPitch,static_cast<const char*>(data)+uint64_t(y)*w*bytesPerPixel,w*bytesPerPixel);
            upload->Unmap(0,nullptr);
            D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;dst.pResource=r.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);barrier(r.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);uploads.push_back(upload);
        }
        return r;
    }
};
#ifndef FSR3_WORKER
int wmain(int argc,wchar_t** argv) {
    try {
        if(argc!=8)throw std::runtime_error("Usage: fsr3-probe <official DLL> <RGBA input> <width> <height> <outWidth> <outHeight> <RGBA output>");
        UINT w=std::stoul(argv[3]),h=std::stoul(argv[4]),ow=std::stoul(argv[5]),oh=std::stoul(argv[6]);
        if(!w||!h||ow<w||oh<h||ow>4096||oh>4096)throw std::runtime_error("Invalid or excessive dimensions");
        std::vector<uint8_t> input(uint64_t(w)*h*4);
        std::ifstream in(argv[2],std::ios::binary);in.read(reinterpret_cast<char*>(input.data()),input.size());
        if(!in||in.peek()!=EOF)throw std::runtime_error("Input RGBA byte count mismatch");
        HMODULE module=LoadLibraryExW(argv[1],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!module)throw std::runtime_error("Official DLL load failed: "+std::to_string(GetLastError()));
        // Keep module loaded until process exit: context/provider objects use it.
        ffxFunctions api{};ffxLoadFunctions(&api,module);
        if(!api.CreateContext||!api.DestroyContext||!api.Query||!api.Dispatch)throw std::runtime_error("Official API exports missing");
        Gpu gpu;
        ffxQueryDescGetVersions versions{};versions.header.type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;versions.createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;versions.device=gpu.device.Get();uint64_t count=0;versions.outputCount=&count;
        apiCheck(api.Query(nullptr,&versions.header),"query provider count");
        if(!count||count>64)throw std::runtime_error("Invalid provider count");
        std::vector<uint64_t> ids(count);std::vector<const char*> names(count);versions.versionIds=ids.data();versions.versionNames=names.data();
        apiCheck(api.Query(nullptr,&versions.header),"query provider names");
        int selected=-1;for(size_t i=0;i<count;i++) {std::cerr<<"Provider "<<ids[i]<<": "<<(names[i]?names[i]:"unknown")<<"\n";if(names[i]&&std::string(names[i]).find("3.1.")!=std::string::npos&&selected<0)selected=int(i);}
        if(selected<0)throw std::runtime_error("No explicit FSR 3.1 provider; do not silently choose FSR2/4");
        ffxCreateBackendDX12Desc backend{};backend.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;backend.device=gpu.device.Get();
        ffxOverrideVersion override{};override.header.type=FFX_API_DESC_TYPE_OVERRIDE_VERSION;override.versionId=ids[selected];override.header.pNext=&backend.header;
        ffxCreateContextDescUpscaleVersion version{};version.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;version.version=FFX_UPSCALER_VERSION;version.header.pNext=&override.header;
        ffxCreateContextDescUpscale create{};create.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;create.header.pNext=&version.header;create.maxRenderSize={w,h};create.maxUpscaleSize={ow,oh};create.flags=FFX_UPSCALE_ENABLE_AUTO_EXPOSURE|FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE|FFX_UPSCALE_ENABLE_DEBUG_CHECKING;create.fpMessage=message;
        ffxContext context=nullptr;apiCheck(api.CreateContext(&context,&create.header,nullptr),"create FSR3 context");
        ffxQueryGetProviderVersion provider{};provider.header.type=FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;apiCheck(api.Query(&context,&provider.header),"verify actual provider");
        if(provider.versionId!=ids[selected])throw std::runtime_error("Provider override mismatch");
        std::vector<float> depth(uint64_t(w)*h,.5f);std::vector<uint16_t> mv(uint64_t(w)*h*2,0);
        auto color=gpu.texture(w,h,DXGI_FORMAT_R8G8B8A8_UNORM,input.data(),4);
        auto dep=gpu.texture(w,h,DXGI_FORMAT_R32_FLOAT,depth.data(),4);
        auto motion=gpu.texture(w,h,DXGI_FORMAT_R16G16_FLOAT,mv.data(),4);
        auto output=gpu.texture(ow,oh,DXGI_FORMAT_R8G8B8A8_UNORM,nullptr,4);
        ffxDispatchDescUpscale dispatch{};dispatch.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;dispatch.commandList=gpu.commands.Get();dispatch.color=ffxApiGetResourceDX12(color.Get());dispatch.depth=ffxApiGetResourceDX12(dep.Get(),FFX_API_RESOURCE_STATE_COMPUTE_READ,FFX_API_RESOURCE_USAGE_DEPTHTARGET);dispatch.motionVectors=ffxApiGetResourceDX12(motion.Get());dispatch.output=ffxApiGetResourceDX12(output.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);dispatch.renderSize={w,h};dispatch.upscaleSize={ow,oh};dispatch.motionVectorScale={float(w),float(h)};dispatch.enableSharpening=true;dispatch.sharpness=.2f;dispatch.frameTimeDelta=1000.f/24;dispatch.preExposure=1;dispatch.reset=true;dispatch.cameraNear=.1f;dispatch.cameraFar=1000;dispatch.cameraFovAngleVertical=1;dispatch.viewSpaceToMetersFactor=1;dispatch.flags=FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        // No synthetic camera jitter: input is decoded video, not jittered rendering.
        apiCheck(api.Dispatch(&context,&dispatch.header),"FSR3 dispatch");
        gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto desc=output->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 bytes;gpu.device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        auto readback=gpu.buffer(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=output.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;
        gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        check(gpu.commands->Close(),"close commands");ID3D12CommandList* lists[]={gpu.commands.Get()};gpu.queue->ExecuteCommandLists(1,lists);gpu.wait();
        if(ffxErrors)throw std::runtime_error("FSR validation errors reported");
        void* mapped=nullptr;D3D12_RANGE range{0,SIZE_T(bytes)};check(readback->Map(0,&range,&mapped),"readback map");
        std::ofstream out(argv[7],std::ios::binary);uint64_t sum=0;
        for(UINT y=0;y<oh;y++) {auto row=static_cast<uint8_t*>(mapped)+fp.Offset+y*fp.Footprint.RowPitch;out.write(reinterpret_cast<char*>(row),ow*4);for(UINT x=0;x<ow;x++)sum+=row[x*4]+row[x*4+1]+row[x*4+2];}
        out.close();D3D12_RANGE empty{0,0};readback->Unmap(0,&empty);if(!out)throw std::runtime_error("Output write failed");
        bool nonBlackInput=false;for(size_t i=0;i<input.size();i+=4)if(input[i]||input[i+1]||input[i+2])nonBlackInput=true;
        if(nonBlackInput&&!sum)throw std::runtime_error("Black GPU output from nonblack input");
        std::string providerName=provider.versionName?provider.versionName:names[selected];
        apiCheck(api.DestroyContext(&context,nullptr),"destroy FSR3 context");
        std::cout<<"{\"status\":\"PASS\",\"adapter\":\""<<escape(gpu.name)<<"\",\"provider\":\""<<escape(providerName)<<"\",\"providerId\":"<<ids[selected]<<",\"input\":["<<w<<","<<h<<"],\"output\":["<<ow<<","<<oh<<"],\"rgbSum\":"<<sum<<",\"scope\":\"Single-frame compatibility only: constant depth, zero motion, zero camera jitter, reset history. Not temporal video validation or player FPS.\"}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FSR3_PROBE_ERROR: "<<error.what()<<"\n";return 1;}
}
#endif
