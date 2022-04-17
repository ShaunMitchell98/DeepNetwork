#pragma once

#ifdef _WIN32
	#define EXPORT __declspec(dllexport)
	#define CUDA
#else	
	#define EXPORT
#endif