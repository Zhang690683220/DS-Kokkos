KOKKOS_PATH = /home/bz186/kokkos
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "writer_compression"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda,force_uvm"
else
CXX = mpicxx
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M
DS_CFLAGS = -I/home/bz186/dataspaces/build_rdma/include
HDF5_CFLAGS = #-I/projects/community/hdf5/1.8.20/gc563/include

OBJ = $(SRC:.cpp=.o)
LIB =
DS_LIBS = -L/home/bz186/dataspaces/build_rdma/lib -ldspaces -ldscommon -ldart   -lm  -lrt   -libverbs  -lrdmacm
HDF5_LIBS = -lhdf5

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(HDF5_LIBS) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(DS_LIBS)  $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(HDF5_LIBS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(DS_CFLAGS) $(DS_LIBS) $(HDF5_CFLAGS)  $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)