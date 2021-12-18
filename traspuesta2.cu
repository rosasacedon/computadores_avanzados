/* Copiar traspuesta de matriz h_a[F][C] en matriz h_b[C][F] aunque el n.º de hebras de 
   los bloques no divida al n.º de componentes de las matrices */
#include <stdio.h>

#define F 25
#define C 43
// matriz original de F filas y C columnas
#define H 16
// bloques de H x H hebras (HxH<=512, capacidad cpto. 1.3)

 __global__ void trspta2(int *dev_a, int *dev_b, int filas, int cols)
{
  __shared__ int s[H*H];	// variable compartida con tantos componentes como hebras tiene un bloque
  int bbx = blockIdx.x * blockDim.x;	// = blockIdx.x * H
  int bby = blockIdx.y * blockDim.y; 	// = blockIdx.y * H 
  int ix = bbx + threadIdx.x;
  int iy = bby + threadIdx.y;  
  int aux;
  int idt = threadIdx.y * blockDim.x + threadIdx.x;	// Identificador de hebra en un bloque
  int idttr = threadIdx.x * blockDim.y + threadIdx.y;

  if ((ix<cols)&&(iy<filas))
  { /* Si S[H][H] es la matriz representada por s, queremos guardar en S la traspuesta de
	   la submatriz de dev_a leída por el bloque de hebras
	   (para, después, colocar S, en el lugar adecuado de dev_b) */
	aux = iy*cols+ix;	// Posición (iy,ix) en la matriz representada por dev_a
	  	/* Dentro de un bloque, cuando idt aumenta en 1 y threadIdx.y no cambia (threadIdx.x aumenta en 1),
	  	   aux aumenta en 1. Esto ocurre en turnos de H veces seguidas ya que 0 <= threadIdx.x < blockDim.x = H.
		   Por tanto, usando aux como índice, hay coalescencia en acceso a memoria global cada H accesos.
		   Como la coalescencia máxima es de 16 accesos, conviene que H=16 */
	s[idttr] = dev_a[aux];	/* S[threadIdx.x][threadIdx.y]= A[iy][ix]
							   (str[idttr] representa S[threadIdx.x][threadIdx.y]) */
  }	  
  /* Ahora debemos copiar s a su lugar en dev_b.
     La esquina superior izda. de s corresponde a la posición (ix,iy) en dev_b con threadIdx.x = threadIdx.y = 0
     Es decir, ix*filas+iy con threadIdx.x = threadIdx.y = 0.
     Esto es, bbx * filas + bby */
  __syncthreads();	
  
  int esqsupizda = bbx * filas + bby;
  /* Si pensamos s como matriz, un recorrido con el índice idt sería un recorrido por filas
     En S seleccionaríamos S[threadIdx.y][threadIdx.x]
     Por tanto, en dev_b el índice debe ser: esqsupizda + threadIdx.y * filas + threadIdx.x */
  
  if (((bbx+threadIdx.y)<cols) && ((bby+threadIdx.x)<filas))
	dev_b[esqsupizda + threadIdx.y * filas + threadIdx.x] = s[idt];
  /* Los límites del if cambian teniendo en cuenta la transposición realizada  */
}


int main(int argc, char** argv)
{
  int h_a[F][C], h_b[C][F];
  int *d_a, *d_b;
  int i, j, aux, size = F * C * sizeof(int);
  dim3 hebrasBloque(H, H); // bloques de H x H hebras
  int numBlf = (F+H-1)/H;  // techo de F/H
  int numBlc = (C+H-1)/H;  // techo de C/H
  dim3 numBloques(numBlc,numBlf);


  // reservar espacio en el device para d_a y d_b
  cudaMalloc((void**) &d_a, size); 
  cudaMalloc((void**) &d_b, size);

  // dar valores a la matriz h_a en la CPU e imprimirlos
  printf("\nMatriz origen\n");
  for (i=0; i<F; i++) {
    for (j=0; j<C; j++) {
      aux = i*C+j;
      h_a[i][j] = aux;
      printf("%d ", aux);
    }
    printf("\n");
  }

  // copiar matriz h_a en d_a
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  
  // llamar al kernel que obtiene en d_b la traspuesta de d_a
  trspta2<<<numBloques, hebrasBloque>>>(d_a, d_b, F, C);

  // copiar matriz d_b en h_b
  cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
 
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i=0; i<F; i++)
    for (j=0; j<C; j++) 
      if (h_a[i][j]!= h_b[j][i]) 
		{printf("error en componente %d %d de matriz de entrada \n", i,j); break;}
		 
// imprimir matriz resultado
  printf("\nMatriz resultado\n");
  for (i=0; i<C; i++) {
    for (j=0; j<F; j++) printf("%d ", h_b[i][j]);
    printf("\n");
  }
  printf("\n");

  cudaFree(d_a); cudaFree(d_b);
  
  return 0;
} 
