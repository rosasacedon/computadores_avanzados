#include <stdio.h>
#include <assert.h>

#define N 100000
#define tb 512	// tamaño bloque

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i=ii; i<N; i+=stride)
	    DC[i] = DA[i] + DB[i];
}

cudaError_t testCuErr(cudaError_t result)
{
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);	// si no se cumple, se aborta el programa
  }
  return result;
}

int main()
{ cudaFree(0);
  int *HA, *HB, *HC, *DA, *DB, *DC;
  int i, dg; int size = N*sizeof(int);

  // HA = (int*)malloc(size); HB = (int*)malloc(size); HC = (int*)malloc(size);
  
  // reservamos espacio en la memoria global del device
  testCuErr(cudaMalloc((void**)&DA, size));
  testCuErr(cudaMalloc((void**)&DB, size));
  testCuErr(cudaMalloc((void**)&DC, size));

  // reservamos espacio en la memoria global del host
  testCuErr(cudaMallocHost((void**)&HA, size));
  testCuErr(cudaMallocHost((void**)&HB, size));
  testCuErr(cudaMallocHost((void**)&HC, size));
     
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  testCuErr(cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice));
  testCuErr(cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice));
      
  dg = (N+tb-1)/tb; if (dg>65535) dg=65535;
  // llamamos al kernel
  VecAdd <<<dg, tb>>>(DA, DB, DC);	// N o más hilos ejecutan el kernel en paralelo
  testCuErr(cudaGetLastError());
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  testCuErr(cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost));
    
  // liberamos la memoria reservada en el device
  testCuErr(cudaFree(DA)); testCuErr(cudaFree(DB)); testCuErr(cudaFree(DC));  
    
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}
 
  // free(HA); free(HB); free(HC);
  cudaFreeHost(HA); cudaFreeHost(HB); cudaFreeHost(HC);
  return 0;
} 
