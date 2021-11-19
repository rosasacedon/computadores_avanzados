#include <stdio.h>

#define N 500

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int i = threadIdx.x;
    DC[i] = DA[i] + DB[i];//se sumara los componentes de un vector y de otro
}

int main()
{ 
  int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;//se tienen que generar de manera dinamica
  int i; int size = N*sizeof(int); //es el tamaño de reserva de espacio
  
  cudaFree(0);

  // reservamos espacio en la memoria global del device
  cudaMalloc((void**)&DA, size);
  cudaMalloc((void**)&DB, size);
  cudaMalloc((void**)&DC, size);
  
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  
  // llamamos al kernel (1 bloque de N hilos)
  VecAdd <<<1, N>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo, los hilos al ser en una dimension se identifican por thread.x
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  
  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
      {printf("error en componente %d\n", i); break;}
    
  return 0;
} 
