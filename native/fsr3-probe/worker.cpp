// Persistent official FSR3/D3D12 video adapter. One bounded frame at a time.
#define FSR3_WORKER
#include "probe.cpp"
#include "motion.hpp"
#include <d3dcompiler.h>
#include <io.h>
#include <fcntl.h>
#include <memory>
#include <array>
struct Request { uint32_t magic,id,width,height,outWidth,outHeight,flags,deltaUs; };
static_assert(sizeof(Request)==32);
class FsrVideo {
    Gpu& gpu; ffxFunctions& api;
    ffxCreateBackendDX12Desc backend{};ffxOverrideVersion override{};ffxCreateContextDescUpscaleVersion version{};ffxCreateContextDescUpscale create{};
    ffxContext context=nullptr;
    ComPtr<ID3D12Resource> color,previous,depth,motion,reactive,output,upload,readback;
    ComPtr<ID3D12RootSignature> root;ComPtr<ID3D12PipelineState> pipeline;ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputFp{},outputFp{};
    uint64_t inputBytes=0,outputBytes=0;UINT width,height,ow,oh;bool hasHistory=false;
    std::chrono::steady_clock::time_point lastFrame{};
public:
    FsrVideo(Gpu& g,ffxFunctions& a,uint64_t provider,const Request& r):gpu(g),api(a),width(r.width),height(r.height),ow(r.outWidth),oh(r.outHeight) {
        backend.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;backend.device=gpu.device.Get();
        override.header.type=FFX_API_DESC_TYPE_OVERRIDE_VERSION;override.versionId=provider;override.header.pNext=&backend.header;
        version.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;version.version=FFX_UPSCALER_VERSION;version.header.pNext=&override.header;
        create.header.type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;create.header.pNext=&version.header;create.maxRenderSize={width,height};create.maxUpscaleSize={ow,oh};create.flags=FFX_UPSCALE_ENABLE_AUTO_EXPOSURE|FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE|FFX_UPSCALE_ENABLE_DEBUG_CHECKING;create.fpMessage=message;
        apiCheck(api.CreateContext(&context,&create.header,nullptr),"FSR3 video context");
        ffxQueryGetProviderVersion actual{};actual.header.type=FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        apiCheck(api.Query(&context,&actual.header),"verify FSR3 video provider");if(actual.versionId!=provider)throw std::runtime_error("FSR3 provider mismatch");
        std::vector<uint8_t> zeros(uint64_t(width)*height*4);std::vector<float> z(uint64_t(width)*height,.5f);
        color=gpu.texture(width,height,DXGI_FORMAT_R8G8B8A8_UNORM,zeros.data(),4);
        previous=gpu.texture(width,height,DXGI_FORMAT_R8G8B8A8_UNORM,zeros.data(),4);
        depth=gpu.texture(width,height,DXGI_FORMAT_R32_FLOAT,z.data(),4);
        motion=gpu.texture(width,height,DXGI_FORMAT_R16G16_FLOAT,nullptr,4);
        reactive=gpu.texture(width,height,DXGI_FORMAT_R8_UNORM,nullptr,1);
        output=gpu.texture(ow,oh,DXGI_FORMAT_R8G8B8A8_UNORM,nullptr,4);
        auto inputDesc=color->GetDesc(),outDesc=output->GetDesc();
        gpu.device->GetCopyableFootprints(&inputDesc,0,1,0,&inputFp,nullptr,nullptr,&inputBytes);
        gpu.device->GetCopyableFootprints(&outDesc,0,1,0,&outputFp,nullptr,nullptr,&outputBytes);
        upload=gpu.buffer(inputBytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        readback=gpu.buffer(outputBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        ComPtr<ID3DBlob> shader,error;
        HRESULT compiled=D3DCompile(MotionShader,sizeof(MotionShader)-1,nullptr,nullptr,nullptr,"main","cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&shader,&error);
        if(FAILED(compiled))throw std::runtime_error(error?static_cast<const char*>(error->GetBufferPointer()):"Motion shader compilation failed");
        D3D12_DESCRIPTOR_RANGE ranges[2]{};ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,2,0,0,0};ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,0,0,2};
        D3D12_ROOT_PARAMETER params[2]{};params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[0].DescriptorTable={2,ranges};params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[1].Constants={0,0,4};
        D3D12_ROOT_SIGNATURE_DESC rootDesc{};rootDesc.NumParameters=2;rootDesc.pParameters=params;
        ComPtr<ID3DBlob> signature;check(D3D12SerializeRootSignature(&rootDesc,D3D_ROOT_SIGNATURE_VERSION_1,&signature,&error),"serialize motion root");
        check(gpu.device->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),IID_PPV_ARGS(&root)),"motion root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};pipelineDesc.pRootSignature=root.Get();pipelineDesc.CS={shader->GetBufferPointer(),shader->GetBufferSize()};check(gpu.device->CreateComputePipelineState(&pipelineDesc,IID_PPV_ARGS(&pipeline)),"motion pipeline");
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};heapDesc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;heapDesc.NumDescriptors=4;heapDesc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;check(gpu.device->CreateDescriptorHeap(&heapDesc,IID_PPV_ARGS(&heap)),"motion heap");
        auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT stride=gpu.device->GetDescriptorHandleIncrementSize(heapDesc.Type);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=DXGI_FORMAT_R8G8B8A8_UNORM;srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;srv.Texture2D.MipLevels=1;
        gpu.device->CreateShaderResourceView(color.Get(),&srv,handle);handle.ptr+=stride;gpu.device->CreateShaderResourceView(previous.Get(),&srv,handle);handle.ptr+=stride;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;uav.Format=DXGI_FORMAT_R16G16_FLOAT;gpu.device->CreateUnorderedAccessView(motion.Get(),nullptr,&uav,handle);handle.ptr+=stride;uav.Format=DXGI_FORMAT_R8_UNORM;gpu.device->CreateUnorderedAccessView(reactive.Get(),nullptr,&uav,handle);
        submit();gpu.uploads.clear();
    }
    ~FsrVideo(){if(context)api.DestroyContext(&context,nullptr);}
    bool matches(const Request& r)const{return width==r.width&&height==r.height&&ow==r.outWidth&&oh==r.outHeight;}
    void submit(){check(gpu.commands->Close(),"video commands close");ID3D12CommandList* list[]={gpu.commands.Get()};gpu.queue->ExecuteCommandLists(1,list);gpu.wait();}
    std::vector<uint8_t> render(const std::vector<uint8_t>& input,bool forceReset,float frameDeltaMs) {
        auto now=std::chrono::steady_clock::now();bool reset=forceReset||!hasHistory||(now-lastFrame>std::chrono::milliseconds(500));
        check(gpu.allocator->Reset(),"video allocator reset");check(gpu.commands->Reset(gpu.allocator.Get(),nullptr),"video commands reset");
        void* mapped=nullptr;D3D12_RANGE empty{0,0};check(upload->Map(0,&empty,&mapped),"video upload map");for(UINT y=0;y<height;y++)memcpy(static_cast<char*>(mapped)+inputFp.Offset+y*inputFp.Footprint.RowPitch,input.data()+uint64_t(y)*width*4,width*4);upload->Unmap(0,nullptr);
        gpu.barrier(color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=upload.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=inputFp;dst.pResource=color.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);gpu.barrier(color.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if(hasHistory){gpu.barrier(motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu.barrier(reactive.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);}
        ID3D12DescriptorHeap* heaps[]={heap.Get()};gpu.commands->SetDescriptorHeaps(1,heaps);gpu.commands->SetComputeRootSignature(root.Get());gpu.commands->SetPipelineState(pipeline.Get());gpu.commands->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());uint32_t dims[]={width,height,reset?0u:1u,0};gpu.commands->SetComputeRoot32BitConstants(1,4,dims,0);gpu.commands->Dispatch((width+63)/64,(height+63)/64,1);
        gpu.barrier(motion.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);gpu.barrier(reactive.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ffxDispatchDescUpscale d{};d.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE;d.commandList=gpu.commands.Get();d.color=ffxApiGetResourceDX12(color.Get());d.depth=ffxApiGetResourceDX12(depth.Get(),FFX_API_RESOURCE_STATE_COMPUTE_READ,FFX_API_RESOURCE_USAGE_DEPTHTARGET);d.motionVectors=ffxApiGetResourceDX12(motion.Get());d.reactive=ffxApiGetResourceDX12(reactive.Get());d.transparencyAndComposition=d.reactive;d.output=ffxApiGetResourceDX12(output.Get(),FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);d.renderSize={width,height};d.upscaleSize={ow,oh};d.motionVectorScale={float(width),float(height)};d.enableSharpening=true;d.sharpness=.2f;d.frameTimeDelta=frameDeltaMs;d.preExposure=1;d.reset=reset;d.cameraNear=.1f;d.cameraFar=1000;d.cameraFovAngleVertical=1;d.viewSpaceToMetersFactor=1;d.flags=FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        apiCheck(api.Dispatch(&context,&d.header),"FSR3 video dispatch");
        gpu.barrier(color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);gpu.barrier(previous.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);gpu.commands->CopyResource(previous.Get(),color.Get());gpu.barrier(previous.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);gpu.barrier(color.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);src={};dst={};src.pResource=output.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=outputFp;gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);submit();
        if(ffxErrors)throw std::runtime_error("FSR3 video validation errors");
        D3D12_RANGE read{0,SIZE_T(outputBytes)};check(readback->Map(0,&read,&mapped),"video readback map");std::vector<uint8_t> result(uint64_t(ow)*oh*4);for(UINT y=0;y<oh;y++)memcpy(result.data()+uint64_t(y)*ow*4,static_cast<char*>(mapped)+outputFp.Offset+y*outputFp.Footprint.RowPitch,ow*4);readback->Unmap(0,&empty);hasHistory=true;lastFrame=now;return result;
    }
};
int wmain(int argc,wchar_t** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Usage: fsr3-worker <official upscaler DLL>");
        _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
        auto module=LoadLibraryExW(argv[1],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);if(!module)throw std::runtime_error("FSR3 DLL unavailable");ffxFunctions api{};ffxLoadFunctions(&api,module);if(!api.Query||!api.CreateContext||!api.Dispatch||!api.DestroyContext)throw std::runtime_error("FSR3 API exports unavailable");
        Gpu gpu;ffxQueryDescGetVersions versions{};versions.header.type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;versions.createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;versions.device=gpu.device.Get();uint64_t count=0;versions.outputCount=&count;apiCheck(api.Query(nullptr,&versions.header),"video provider count");if(!count||count>64)throw std::runtime_error("Invalid provider count");std::vector<uint64_t> ids(count);std::vector<const char*> names(count);versions.versionIds=ids.data();versions.versionNames=names.data();apiCheck(api.Query(nullptr,&versions.header),"video provider list");size_t selected=count;for(size_t i=0;i<count;i++)if(names[i]&&std::string(names[i]).find("3.1.")!=std::string::npos){selected=i;break;}if(selected==count)throw std::runtime_error("No official FSR3.1 provider");
        std::cout<<"{\"backend\":\"d3d12\",\"provider\":\""<<escape(names[selected])<<"\",\"gpu\":\""<<escape(gpu.name)<<"\"}\n"<<std::flush;
        std::unique_ptr<FsrVideo> renderer;bool commandsClosed=false;
        while(true) {
            Request r{};std::cin.read(reinterpret_cast<char*>(&r),sizeof(r));if(std::cin.gcount()==0&&std::cin.eof())break;if(!std::cin)throw std::runtime_error("Truncated FSR3 frame header");
            if(r.magic!=0x31435341||r.flags>1||r.deltaUs<1000||r.deltaUs>250000||r.width<40||r.height<40||r.width%4||r.height%4||r.width>1920||r.height>1920||uint64_t(r.width)*r.height>1920*1080||r.outWidth<r.width||r.outHeight<r.height||r.outWidth%2||r.outHeight%2||r.outWidth>3840||r.outHeight>3840||uint64_t(r.outWidth)*r.outHeight>3840*2160)throw std::runtime_error("Invalid FSR3 frame dimensions/flags");
            std::vector<uint8_t> input(uint64_t(r.width)*r.height*4);std::cin.read(reinterpret_cast<char*>(input.data()),input.size());if(!std::cin)throw std::runtime_error("Truncated FSR3 pixels");
            if(!renderer||!renderer->matches(r)){renderer.reset();if(commandsClosed){check(gpu.allocator->Reset(),"configure allocator");check(gpu.commands->Reset(gpu.allocator.Get(),nullptr),"configure commands");}renderer=std::make_unique<FsrVideo>(gpu,api,ids[selected],r);commandsClosed=true;}
            auto output=renderer->render(input,r.flags==1,float(r.deltaUs)/1000);uint32_t header[]={0x31435341,r.id,r.outWidth,r.outHeight,r.outWidth,r.outHeight};std::cout.write(reinterpret_cast<const char*>(header),sizeof(header));std::cout.write(reinterpret_cast<const char*>(output.data()),output.size());std::cout.flush();if(!std::cout)break;
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"FSR3_WORKER_ERROR: "<<e.what()<<"\n";return 1;}
}
