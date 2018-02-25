/**
UU-INFOMCV 2018
Assignment 2 - Voxel Reconstruction

Satwiko Wirawan Indrawanto - 6201539
Basar Oguz - 6084990
*/

#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;

int main(int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
