/* ---------------------------------------------------
   My Hello world for CUDA programming
   --------------------------------------------------- */

   #include <stdio.h>        // C programming header file
   #include <unistd.h>       // C programming header file
                             // cude.h is automatically included by nvcc...
   
   /* ------------------------------------
      Your first kernel (= GPU function)
      ------------------------------------ */
   __global__ void hello( )
   {
      printf("Hello World !\n");
   }
   
   int main()
   {
      /* ------------------------------------
         Call the hello( ) kernel function
         ------------------------------------ */
      hello<<< 1, 4 >>>( );
      sleep(1);   // Necessary to give time to let GPU threads run !!!

      printf("I am the CPU: Hello World ! \n");
   
      return 0;
   }