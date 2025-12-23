Trilinos/PETSc Sparsity Pattern (Dario:maybe done idk) 

Remove any zero from the Sparsity Pattern

Preconditioner

Check feasibility of MatrixFree

Squared finite elements (Ricky)

check also the uneven dofs split

DONE config file: dealii function for config file ParameterHandler
DONE Triangulation by n dofs
profiler su vscode e su cluster
Sparsity pattern to check
Refinement
leggere dealii 32?

GRAFICI
CHECK DIFFERENZA 3.3 e 4.0
ripulire ultima volta il codice
leggere il report
controllare il report
commentare il codice
fare grafici scalabilità

1) STRONG NO REFINE
partiamo da pari, poi male che vada andiamo a potenze di due
32 può essere decente? sennò 64

2) STRONG CON REFINE
partiamo da pari, poi male che vada andiamo a potenze di due
si parte da 8 comunque

3) WEAK NO REFINE
per ora abbiamo 1 4 16 processori e 1 - 4 - 16 * 8 come grid

4)Refined e non refined allo stesso livello

4)Iterativo versus diretto ?
2D e 3D, penso basti poco 