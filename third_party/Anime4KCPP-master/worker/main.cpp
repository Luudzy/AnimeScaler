#include <AC/Core.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kMagic = 0x31435341u;
constexpr std::uint32_t kResponseMagic = 0x33435341u;
struct Header { std::uint32_t magic, frame, width, height, outWidth, outHeight; };

bool read_exact(void* target, std::size_t bytes) {
    auto* buffer = std::cin.rdbuf();
    auto* output = static_cast<char*>(target);
    std::size_t received = 0;
    while (received < bytes) {
        const auto count = buffer->sgetn(output + received,
                                         static_cast<std::streamsize>(bytes - received));
        if (count <= 0) return false;
        received += static_cast<std::size_t>(count);
    }
    return true;
}

bool write_exact(const void* source, std::size_t bytes) {
    std::cout.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
    std::cout.flush();
    return std::cout.good();
}

bool valid(const Header& h) {
    return h.magic == kMagic && h.width >= 40 && h.height >= 40 &&
        h.width % 4 == 0 && h.height % 4 == 0 &&
        static_cast<std::uint64_t>(h.width) * h.height <= 1920ull * 1080ull &&
        h.outWidth >= 2 && h.outHeight >= 2 && h.outWidth % 2 == 0 && h.outHeight % 2 == 0 &&
        static_cast<std::uint64_t>(h.outWidth) * h.outHeight <= 1920ull * 1080ull * 4ull;
}
}

int main(int argc, char** argv) {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    const std::string model = argc > 1 ? argv[1] : "arnet-f8b8";
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    auto processor = ac::core::Processor::create("opencl", device, model.c_str());
    if (!processor || !processor->ok()) {
        std::cerr << "Anime4KCPP OpenCL initialization failed: "
                  << (processor ? processor->error() : "no processor") << std::endl;
        return 2;
    }

    Header header{};
    while (read_exact(&header, sizeof(header))) {
        if (!valid(header)) {
            std::cerr << "Invalid frame header" << std::endl;
            return 3;
        }
        const std::size_t inputBytes = static_cast<std::size_t>(header.width) * header.height * 4;
        std::vector<std::uint8_t> input(inputBytes);
        if (!read_exact(input.data(), input.size())) return 0;
        ac::core::Image source(header.width, header.height, 4, ac::core::Image::UInt8,
                               input.data(), header.width * 4);
        ac::core::Image native = processor->process(source, 2.0);
        if (native.empty()) {
            std::cerr << "Anime4KCPP returned an empty frame: " << processor->error() << std::endl;
            return 4;
        }
        ac::core::Image output;
        if (native.width() == static_cast<int>(header.outWidth) && native.height() == static_cast<int>(header.outHeight)) {
            output = std::move(native);
        } else {
            output.create(header.outWidth, header.outHeight, 4, ac::core::Image::UInt8);
            ac::core::resize(native, output, 0, 0, ac::core::RESIZE_BICUBIC_0_60);
        }
        Header response{kResponseMagic, header.frame, header.width, header.height,
                        header.outWidth, header.outHeight};
        const std::size_t outputBytes = static_cast<std::size_t>(header.outWidth) * header.outHeight * 4;
        std::vector<std::uint8_t> packed(outputBytes);
        output.to(packed.data(), header.outWidth * 4);
        if (!write_exact(&response, sizeof(response)) || !write_exact(packed.data(), packed.size())) return 0;
    }
    return 0;
}
