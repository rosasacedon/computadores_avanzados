#include <stdio.h>

#define N 65535*512+1

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int ii = (blockIdx.x*blockDim.x)+threadIdx.x;
	int stride=blockDim.x * gridDim.x;
	for(int i=ii;i<N; i+=stride){
		DC[i] = DA[i] + DB[i];//se sumara los componentes de un vector y de otro
	}
}

int main()
{ int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;//se tienen que generar de manera dinamica
  int i; int size = N*sizeof(int); //es el tamaño de reserva de espacio
  
  int dg=min((N/512),65535);
  cudaFree(0);
  cudaError_t tester;

  // reservamos espacio en la memoria global del device
  tester=cudaMalloc((void**)&DA, size);
  if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  tester=cudaMalloc((void**)&DB, size);
  if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  tester=cudaMalloc((void**)&DC, size);
  if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  tester=cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  tester=cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  
  // llamamos al kernel (1 bloque de N hilos)
  
  VecAdd <<<dg, 512>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo, los hilos al ser en una dimension se identifican por thread.x
  tester = cudaGetLastError();
 if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  tester=cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
   if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  // liberamos la memoria reservada en el device
  tester=cudaFree(DA); cudaFree(DB); cudaFree(DC);  
   if(tester!=cudaSuccess){
    printf("Error en cuda %s", cudaGetErrorString(tester));
    exit(0);
  }
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
    {printf("error en componente %d\n", i); break;}
    
  return 0;
} 
