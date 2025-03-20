# Compiler
NVCC = "D:/External Installations/bin/nvcc.exe"

# Compute capability
CUDA_ARCH = -gencode=arch=compute_52,code=\"sm_52,compute_52\"

# Include directories
INCLUDES = -I"./include" \
           -I"D:/External Installations/include" \
           -I"D:/External_Apps/GLEW/include" \
           -I"D:/External_Apps/GLFW/include" \
           -I"D:/External_Apps/JSON/json/include" \
           -I"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/include" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/winrt" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/cppwinrt"

# Preprocessor macros
DEFINES = -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS

# Host compiler flags with spaces
XCOMPILER_FLAGS = "/std:c++17 /EHsc /W3 /nologo /Od /FS /Zi /RTC1 /MDd"

# Define the host compiler
CCBIN = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe"

# NVCC compile flags
NVCC_COMPILE_FLAGS = $(CUDA_ARCH) -G -g -std=c++17 $(INCLUDES) $(DEFINES) \
                     -Xcompiler $(XCOMPILER_FLAGS) \
                     -cudart static --use-local-env --machine 64 --compile -x cu \
                     -ccbin $(CCBIN)

# Linker flags (use -Xlinker and properly quote paths with spaces)
LINKER_FLAGS = -Xlinker /LIBPATH:\"D:/External_Apps/GLEW/lib\" \
               -Xlinker /LIBPATH:\"D:/External_Apps/GLFW/lib\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/VC/Tools/MSVC/14.41.34120/lib/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/ucrt/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/um/x64\" \
               -lglew32 -lopengl32 -lglfw3 -luser32 -lgdi32 -lwinmm

# NVCC link flags
NVCC_LINK_FLAGS = $(CUDA_ARCH) -G -g $(INCLUDES) $(DEFINES) \
                  -Xcompiler $(XCOMPILER_FLAGS) \
                  -cudart static --use-local-env --machine 64 \
                  -ccbin $(CCBIN)

# Directories
SRCDIR = src
OBJDIR = obj

# Targets
TARGET = raytracer.exe

# Shared source files
SHARED_SRC = $(wildcard $(SRCDIR)/math_utils.cpp) \
             $(wildcard $(SRCDIR)/opengl_utils.cpp) \
             $(wildcard $(SRCDIR)/obj_loader.cpp) \
             $(wildcard $(SRCDIR)/bvh.cpp) \
             $(wildcard $(SRCDIR)/ray.cpp) 

SHARED_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.obj,$(SHARED_SRC:.cu=.cpp)) \
             $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.obj,$(SHARED_SRC:.cpp=.cu))

# Ray Tracer specific source files
RAYTRACER_SRC = $(SRCDIR)/raytracer.cu \
                $(SRCDIR)/main.cpp

RAYTRACER_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.obj,$(RAYTRACER_SRC:.cu=.cpp)) \
                $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.obj,$(RAYTRACER_SRC:.cpp=.cu))

# Build ray tracer executable
$(TARGET): $(RAYTRACER_OBJ) $(SHARED_OBJ)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Compile Rules ###

# Compile shared source files
$(OBJDIR)/%.obj: $(SRCDIR)/%.cpp
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Run target
run: $(TARGET)

# Clean all build files
clean:
	rm -f $(OBJDIR)/*.obj
	rm -f $(TARGET)
	rm -f $(TARGET:.exe=.exp) $(TARGET:.exe=.lib) $(TARGET:.exe=.pdb)
	rm -f vc*.pdb

# Phony targets
.PHONY: all clean run