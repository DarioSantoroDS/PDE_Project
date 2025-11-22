#ifndef FluidStructureInteractionProblem
#define FluidStructureInteractionProblem

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <petscviewer.h> // Essential for the viewer commands

#define FORCE_USE_OF_TRILINOS


#include <fstream>
#include <iostream>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


using namespace dealii;

class FluidStructureProblem
{
public:
  static constexpr unsigned int dim = 2;

  FluidStructureProblem(const unsigned int stokes_degree,
                        const unsigned int elasticity_degree)
    : stokes_degree(stokes_degree)
    , elasticity_degree(elasticity_degree)
    , triangulation(MPI_COMM_WORLD, Triangulation<dim>::maximum_smoothing)
    , stokes_fe(FE_Q<dim>(stokes_degree + 1),
                dim,
                FE_Q<dim>(stokes_degree),
                1,
                FE_Nothing<dim>(),
                dim)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , elasticity_fe(FE_Nothing<dim>(),
                    dim,
                    FE_Nothing<dim>(),
                    1,
                    FE_Q<dim>(elasticity_degree),
                    dim)
    , dof_handler(triangulation)
    , viscosity(2)
    , lambda(1)
    , mu(1)
    , pcout(std::cout, mpi_rank == 0)
  {
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(elasticity_fe);
  }
  void
  make_grid();
  void
  setup_dofs();
  void
  assemble_system();
  // void
  // solve();
  void
  solve_iterative();
  void
  output_results(const unsigned int refinement_cycle) const;
#ifdef DEBUG
  void
  output_matrix() const;
#endif

  // void refine_mesh();

  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues()
      : Function<dim>(dim + 1 + dim)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      Assert(component < this->n_components,
             ExcIndexRange(component, 0, this->n_components));

      if (component == dim - 1)
        switch (dim)
          {
            case 2:
              return std::sin(numbers::PI * p[0]);
            case 3:
              return std::sin(numbers::PI * p[0]) *
                     std::sin(numbers::PI * p[1]);
            default:
              Assert(false, ExcNotImplemented());
          }

      return 0;
    }

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = StokesBoundaryValues::value(p, c);
    }
  };
  class BlockILUPreconditioner
  {
  public:
    BlockILUPreconditioner() = default;

    void
    initialize(
      const TrilinosWrappers::BlockSparseMatrix               &matrix,
      const TrilinosWrappers::PreconditionILU::AdditionalData &additional_data =
        TrilinosWrappers::PreconditionILU::AdditionalData())
    {
      // 1. Resize the vector of preconditioners to match the number of blocks
      preconditioners.resize(matrix.n_block_rows());

      // 2. Initialize an ILU preconditioner for each diagonal block
      for (unsigned int i = 0; i < matrix.n_block_rows(); ++i)
        {
          preconditioners[i] =
            std::make_shared<TrilinosWrappers::PreconditionILU>();

          // Initialize with the specific sub-matrix (A_00, A_11, etc.)
          preconditioners[i]->initialize(matrix.block(i, i), additional_data);
        }
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Apply the local ILU preconditioners to each vector block independently
      for (unsigned int i = 0; i < preconditioners.size(); ++i)
        {
          // dst_i = ILU(A_ii)^-1 * src_i
          preconditioners[i]->vmult(dst.block(i), src.block(i));
        }
    }

  private:
    std::vector<std::shared_ptr<TrilinosWrappers::PreconditionILU>>
      preconditioners;
  };


  class PreconditionIdentityi
  {
  public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }

  protected:
  };

  class PreconditionBlockTriangular
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
               const LA::MPI::SparseMatrix &pressure_mass_, // pressurematrix(1,1)
               const LA::MPI::SparseMatrix &B_, // A(1,0)
               const LA::MPI::SparseMatrix &D1_, // A(2,0)
               const LA::MPI::SparseMatrix &D2_,  // A(2,1)
               const LA::MPI::SparseMatrix &solid_matrix_ // A(2,2)
    )                  
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;
      D1                 = &D1_;
      D2                 = &D2_;
      solid_matrix       = &solid_matrix_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
      preconditioner_solid.initialize(solid_matrix_,
                                     TrilinosWrappers::PreconditionSSOR::
                                       AdditionalData(1.0
                                        , 1 //added by copilot
                                      ));
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000, 
                                            1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      tmpStokes.reinit(src.block(1));
      B->vmult(tmpStokes, dst.block(0));
      tmpStokes.sadd(-1.0, src.block(1));

      SolverControl                           solver_control_pressure(1000,
                                            1e-2 * src.block(1).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmpStokes,
                               preconditioner_pressure);

      tmpSolid.reinit(src.block(2));
      D1->vmult(tmpSolid, dst.block(0));
      D2->vmult_add(tmpSolid, dst.block(1));
      tmpSolid.sadd(-1.0, src.block(2));

      SolverControl                           solver_control_solid(1000,
                                            1e-2 * src.block(2).l2_norm());

      preconditioner_solid.vmult(dst.block(2), tmpSolid);
    }

  protected:
    // Velocity stiffness matrix.
    const LA::MPI::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const LA::MPI::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const LA::MPI::SparseMatrix *B;

    // D1 matrix.
    const LA::MPI::SparseMatrix *D1;

    // D2 matrix.
    const LA::MPI::SparseMatrix *D2;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionSSOR preconditioner_solid;

    // Solid matrix.
    const LA::MPI::SparseMatrix *solid_matrix;

    // Temporary vector stokes
    mutable LA::MPI::Vector tmpStokes;

    // Temporary vector solid
    mutable LA::MPI::Vector tmpSolid;
  };

private:
  enum
  {
    fluid_domain_id,
    solid_domain_id
  };

  static bool
  cell_is_in_fluid_domain(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }

  static bool
  cell_is_in_solid_domain(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }

  void
  set_active_fe_indices();
  void
  assemble_interface_term(
    const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
    const FEFaceValuesBase<dim>          &stokes_fe_face_values,
    std::vector<Tensor<1, dim>>          &elasticity_phi,
    std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
    std::vector<double>                  &stokes_phi_p,
    FullMatrix<double>                   &local_interface_matrix) const;

  const unsigned int stokes_degree;
  const unsigned int elasticity_degree;
  // Number of MPI processes.
  parallel::distributed::Triangulation<dim> triangulation;


  FESystem<dim>      stokes_fe;
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // MPI_Comm           mpi_communicator;


  FESystem<dim>         elasticity_fe;
  hp::FECollection<dim> fe_collection;
  DoFHandler<dim>       dof_handler;
  const double          viscosity;
  const double          lambda;
  const double          mu; 

public:
  ConditionalOStream pcout;

private:
  AffineConstraints<double> constraints;

  SparsityPattern       sparsity_pattern;
  LA::MPI::BlockSparseMatrix system_matrix;
  LA::MPI::BlockSparseMatrix pressure_mass;
  LA::MPI::BlockVector       solution;
  LA::MPI::BlockVector       locally_relevant_solution;
  LA::MPI::BlockVector       system_rhs;

  IndexSet              locally_owned_dofs;
  IndexSet              locally_relevant_dofs;
  std::vector<IndexSet> block_owned_dofs;
  std::vector<IndexSet> block_relevant_dofs;
};

#endif