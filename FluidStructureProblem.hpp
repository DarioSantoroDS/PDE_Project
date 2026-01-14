#ifndef FluidStructureInteractionProblem
#define FluidStructureInteractionProblem

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>

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
#define ALTERNATIVE_PATTERN
#define ITERATIVE_SOLVER
// #define DIRECT_SOLVER

#include <fstream>
#include <iostream>

#if !defined(ITERATIVE_SOLVER) && !defined(DIRECT_SOLVER)
#  error Either ITERATIVE_SOLVER or DIRECT_SOLVER must be defined.
#endif

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

class ParameterReader : public Subscriptor
{
public:
  ParameterReader(ParameterHandler &);
  void
  read_parameters(const std::string &);

private:
  void
                    declare_parameters();
  ParameterHandler &prm;
};


class FluidStructureProblem
{
public:

  static constexpr unsigned int dim = 2;

  class ExactSolution_u : public Function<dim>
{
public:
  ExactSolution_u()
    : Function<dim>(dim + 1 + dim) // 5 componenti
  {}

  // ======================
  // valore singola componente
  // ======================
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 0)
      return (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
             * y * (-1.0 + 4.0*y*y);

    if (component == 1)
      return -(1.0/64.0) * x * (-1.0 + 16.0*x*x)
             * std::pow(1.0 - 4.0*y*y, 2);

    return 0.0;
  }

  // ======================
  // valore vettoriale
  // ======================
  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }

  // ======================
  // gradiente
  // grad[i][j] = ∂ u_i / ∂ x_j
  // ======================
  virtual void
  vector_gradient(const Point<dim> &p,
           std::vector<Tensor<1, dim>> &grad) const override
  {
    AssertDimension(grad.size(), dim);

    const double x = p[0];
    const double y = p[1];

    // ∂u0/∂x
    grad[0][0] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x) * y * (-1.0 + 4.0*y*y);

    // ∂u0/∂y
    grad[0][1] =
      (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
      * (-1.0 + 12.0*y*y);

    // ∂u1/∂x
    grad[1][0] =
      (1.0/64.0) * (1.0 - 48.0*x*x)
      * std::pow(1.0 - 4.0*y*y, 2);

    // ∂u1/∂y
    grad[1][1] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x)
      * y * (1.0 - 4.0*y*y);
  }
};

class ExactSolution_onlyu : public Function<dim>
{
public:
  ExactSolution_onlyu()
    : Function<dim>(dim) // 2 componenti
  {}

  // ======================
  // valore singola componente
  // ======================
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 0)
      return (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
             * y * (-1.0 + 4.0*y*y);

    if (component == 1)
      return -(1.0/64.0) * x * (-1.0 + 16.0*x*x)
             * std::pow(1.0 - 4.0*y*y, 2);

    return 0.0;
  }

  // ======================
  // valore vettoriale
  // ======================
  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }

  virtual void
  vector_gradient(const Point<dim> &p,
           std::vector<Tensor<1, dim>> &grad) const override
  {
    AssertDimension(grad.size(), dim);

    const double x = p[0];
    const double y = p[1];

    // ∂u0/∂x
    grad[0][0] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x) * y * (-1.0 + 4.0*y*y);

    // ∂u0/∂y
    grad[0][1] =
      (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
      * (-1.0 + 12.0*y*y);

    // ∂u1/∂x
    grad[1][0] =
      (1.0/64.0) * (1.0 - 48.0*x*x)
      * std::pow(1.0 - 4.0*y*y, 2);

    // ∂u1/∂y
    grad[1][1] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x)
      * y * (1.0 - 4.0*y*y);
  }
};

  class ExactSolution_d : public Function<dim>
{
public:
  ExactSolution_d()
    : Function<dim>(dim + 1 + dim) // 5 componenti
  {}

  // ======================
  // valore singola componente
  // ======================
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 3) // d0
      return (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
             * y * (-1.0 + 4.0*y*y);

    if (component == 4) // d1
      return -(1.0/64.0) * x * (-1.0 + 16.0*x*x)
             * std::pow(1.0 - 4.0*y*y, 2);

    return 0.0;
  }

  // ======================
  // valore vettoriale
  // ======================
  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }

  // ======================
  // gradiente
  // grad[i][j] = ∂(componente i)/∂x_j
  // ======================
  virtual void
  vector_gradient(const Point<dim> &p,
                  std::vector<Tensor<1, dim>> &grad) const override
  {
    // inizializza tutto a zero
    for (auto &g : grad)
      g = 0.0;

    const double x = p[0];
    const double y = p[1];

    // ∂d0/∂x
    grad[3][0] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x) * y * (-1.0 + 4.0*y*y);

    // ∂d0/∂y
    grad[3][1] =
      (1.0/256.0) * std::pow(1.0 - 16.0*x*x, 2)
      * (-1.0 + 12.0*y*y);

    // ∂d1/∂x
    grad[4][0] =
      (1.0/64.0) * (1.0 - 48.0*x*x)
      * std::pow(1.0 - 4.0*y*y, 2);

    // ∂d1/∂y
    grad[4][1] =
      (1.0/4.0) * x * (-1.0 + 16.0*x*x)
      * y * (1.0 - 4.0*y*y);
  }
};

class ExactSolution_p : public Function<dim>
{
public:
  ExactSolution_p()
    : Function<dim>(dim + 1 + dim) // 5 componenti
  {}

  // ======================
  // valore singola componente
  // ======================
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 2) // pressione
      return (x - 1.0/4.0)*(x + 1.0/4.0)
             * (y - 1.0/2.0)*(y + 1.0/2.0);

    return 0.0;
  }

  // ======================
  // valore vettoriale
  // ======================
  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }

  // ======================
  // gradiente vettoriale
  // grad[i][j] = ∂(componente i)/∂x_j
  // ======================
  virtual void
  vector_gradient(const Point<dim> &p,
                  std::vector<Tensor<1, dim>> &grad) const override
  {
    // tutto a zero
    for (auto &g : grad)
      g = 0.0;

    const double x = p[0];
    const double y = p[1];

    // ∂p/∂x  → componente 2
    grad[2][0] =
      (x - 1.0/4.0)*(y - 1.0/2.0)*(y + 1.0/2.0)
    + (x + 1.0/4.0)*(y - 1.0/2.0)*(y + 1.0/2.0);

    // ∂p/∂y  → componente 2
    grad[2][1] =
      (x - 1.0/4.0)*(x + 1.0/4.0)*(y - 1.0/2.0)
    + (x - 1.0/4.0)*(x + 1.0/4.0)*(y + 1.0/2.0);
  }
};

class ExactForce_f : public Function<dim>
{
public:
  ExactForce_f(const double viscosity)
    : Function<dim>(dim + 1 + dim)
    , nu(viscosity)
  {}

  // ======================
  // valore singola componente
  // ======================
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 0)
      return x * (-0.5 + 2.0*y*y)
           - (11.0/32.0) * y * nu
           - 24.0 * std::pow(x,4) * y * nu
           + std::pow(y,3) * nu
           + 3.0 * x*x * y * (5.0 - 16.0*y*y) * nu;

    if (component == 1)
      return (-1.0/8.0 + 2.0*x*x) * y
           + (7.0/4.0) * x * nu
           - 4.0 * std::pow(x,3) * nu
           + 3.0 * x * (-5.0 + 16.0*x*x) * y*y * nu
           + 24.0 * x * std::pow(y,4) * nu;

    return 0.0;
  }

  // ======================
  // valore vettoriale
  // ======================
  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }

private:
  const double nu;
};



class ExactForce_g : public Function<dim>
{
public:
  ExactForce_g(const double mu_)
    : Function<dim>(dim + 1 + dim) // 5 componenti
    , mu(mu_)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 3)
      return (-(11.0/32.0) + 15.0*x*x - 24.0*std::pow(x,4)) * y * mu
             + std::pow(y,3) * (mu - 48.0 * x*x * mu);

    if (component == 4)
      return 1.0/4.0 * x * (7.0 - 60.0*y*y + 96.0*std::pow(y,4)
                            + 16.0 * x*x * (-1.0 + 12.0*y*y)) * mu;

    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }
  private:
    const double mu;
};



class ExactNeumann_hRight : public Function<dim>
{
public:
  ExactNeumann_hRight(const double viscosity_)
    : Function<dim>(dim + 1 + dim) // 5 componenti
    , nu(viscosity_)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 0)
      return 1.0/64.0 * (-1.0 + 16.0*x*x) * (-1.0 + 4.0*y*y) * (-1.0 + 16.0*x*y*nu);

    if (component == 1)
      return 1.0/512.0 * (4.0 * (1.0 - 48.0*x*x) * std::pow(1.0 - 4.0*y*y,2)
                          + std::pow(1.0 - 16.0*x*x,2) * (-1.0 + 12.0*y*y)) * nu;

    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }
  private:
    const double nu; 
};

class ExactNeumann_hLeft : public Function<dim>
{
public:
  ExactNeumann_hLeft(const double viscosity_)
    : Function<dim>(dim + 1 + dim) // 5 componenti
    , nu(viscosity_)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];


    if (component == 0)
      return -(1.0/64.0) * (-1.0 + 16.0*x*x) * (-1.0 + 4.0*y*y) * (-1.0 + 16.0*x*y*nu);

    if (component == 1)
      return -(1.0/512.0) * (4.0 * (1.0 - 48.0*x*x) * std::pow(1.0 - 4.0*y*y,2)
                             + std::pow(1.0 - 16.0*x*x,2) * (-1.0 + 12.0*y*y)) * nu;

    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    AssertDimension(values.size(), this->n_components);

    for (unsigned int c = 0; c < this->n_components; ++c)
      values[c] = value(p, c);
  }
private:
    const double nu;
};

  FluidStructureProblem(const unsigned int stokes_degree,
                        const unsigned int elasticity_degree,
                        ParameterHandler  &param)
    : stokes_degree(stokes_degree)
    , elasticity_degree(elasticity_degree)
    , prm(param)
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
    , viscosity(1)
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
#ifdef DIRECT_SOLVER
  void
  solve();
#endif
#ifdef ITERATIVE_SOLVER
  void
  solve_iterative();
#endif
  void
  output_results(const unsigned int refinement_cycle) const;
#ifdef DEBUG
  void
  output_matrix() const;
#endif

  double
  compute_velocity_error(const VectorTools::NormType &norm_type) const;

  void
  refine_mesh();

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



  class PreconditionBlockTriangular
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(
      const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
      const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
      const LA::MPI::SparseMatrix &B_,                  // A(1,0)
      const LA::MPI::SparseMatrix &D1_,                 // A(2,0)
      const LA::MPI::SparseMatrix &D2_,                 // A(2,1)
      const LA::MPI::SparseMatrix &solid_matrix_        // A(2,2)
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
      preconditioner_solid.initialize(
        solid_matrix_
        // , TrilinosWrappers::PreconditionSSOR::AdditionalData(
        //   1.0
        //   // , 1 //i dont think is useful this
        //   )
      );
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
      // std::cout << "  " << solver_control_velocity.last_step() << " CG1
      // iterations"
      //       << std::endl;

      tmpStokes.reinit(src.block(1));
      B->vmult(tmpStokes, dst.block(0));
      tmpStokes.sadd(-1.0, src.block(1));

      SolverControl solver_control_pressure(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmpStokes,
                               preconditioner_pressure);
      // std::cout << "  " << solver_control_pressure.last_step() << " CG2
      // iterations"
      //       << std::endl;

      tmpStokes.reinit(src.block(2));
      D1->vmult(tmpStokes, dst.block(0));
      D2->vmult_add(tmpStokes, dst.block(1));
      tmpStokes.sadd(-1.0, src.block(2));



      // preconditioner_solid.vmult(dst.block(2), tmpStokes);

      SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_solid(
        solver_control_solid);

      solver_cg_solid.solve(*solid_matrix,
                            dst.block(2),
                            tmpStokes,
                            preconditioner_solid);
      // std::cout << "  " << solver_control_solid.last_step() << " CG3
      // iterations"
      //       << std::endl;
    }

  protected:
    // Velocity stiffness matrix.
    const LA::MPI::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionAMG preconditioner_velocity;

    // Pressure mass matrix.
    const LA::MPI::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionAMG preconditioner_pressure;

    // B matrix.
    const LA::MPI::SparseMatrix *B;

    // D1 matrix.
    const LA::MPI::SparseMatrix *D1;

    // D2 matrix.
    const LA::MPI::SparseMatrix *D2;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionAMG preconditioner_solid;

    // Solid matrix.
    const LA::MPI::SparseMatrix *solid_matrix;

    // Temporary vector stokes
    mutable LA::MPI::Vector tmpStokes;

    // // Temporary vector solid
    // mutable LA::MPI::Vector tmpStokes;
  };

  class PreconditionBlockTriangularSimple
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(
      const LA::MPI::SparseMatrix &velocity_stiffness_, // A(0,0)
      const LA::MPI::SparseMatrix &pressure_mass_,      // pressurematrix(1,1)
      const LA::MPI::SparseMatrix &B_,                  // A(1,0)
      const LA::MPI::SparseMatrix &D1_,                 // A(2,0)
      const LA::MPI::SparseMatrix &D2_,                 // A(2,1)
      const LA::MPI::SparseMatrix &solid_matrix_        // A(2,2)
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
      preconditioner_solid.initialize(
        solid_matrix_
        // , TrilinosWrappers::PreconditionSSOR::AdditionalData(
        //   1.0
        //   // , 1 //i dont think is useful this
        //   )
      );
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      //--------------- SIMPLE velocity prediction: u* = A^{-1} rhs_u
      //---------------
      SolverControl solver_control_u(1000, 1e-2 * src.block(0).l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_u(solver_control_u);

      solver_u.solve(*velocity_stiffness,
                     dst.block(0),
                     src.block(0),
                     preconditioner_velocity);

      // u* is now stored in dst.block(0)

      //--------------- Compute pressure rhs: g = B u* - rhs_p
      //----------------------
      tmpStokes.reinit(src.block(1));
      B->vmult(tmpStokes, dst.block(0));  // tmpStokes = B*u*
      tmpStokes.sadd(-1.0, src.block(1)); // tmpStokes = B*u* - rhs_p

      //--------------- SIMPLE Schur complement solve: p = S^{-1} g
      //-----------------
      // We approximate S^{-1} ≈ (pressure_mass)^{-1}
      SolverControl solver_control_S(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);

      solver_S.solve(*pressure_mass,
                     dst.block(1),
                     tmpStokes,
                     preconditioner_pressure);

      // dst.block(1) = p

      //--------------- SIMPLE velocity correction: u = u* - A^{-1} B^T p
      //-----------
      tmpStokes.reinit(src.block(0));
      B->Tvmult(tmpStokes, dst.block(1)); // tmpStokes = B^T p

      TrilinosWrappers::MPI::Vector AuInv(tmpStokes);
      SolverControl solver_control_Acorr(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_Acorr(
        solver_control_Acorr);

      solver_Acorr.solve(*velocity_stiffness,
                         AuInv,
                         tmpStokes,
                         preconditioner_velocity);

      dst.block(0).sadd(1.0, -1.0, AuInv); // u = u* − A^{-1} B^T p

      //--------------- Solid block stays unchanged
      //---------------------------------
      tmpStokes.reinit(src.block(2));
      D1->vmult(tmpStokes, dst.block(0));
      D2->vmult_add(tmpStokes, dst.block(1));
      tmpStokes.sadd(-1.0, src.block(2));
      SolverControl solver_control_solid(1000, 1e-2 * tmpStokes.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_solid(
        solver_control_solid);

      solver_solid.solve(*solid_matrix,
                         dst.block(2),
                         tmpStokes,
                         preconditioner_solid);
    }


  protected:
    // Velocity stiffness matrix.
    const LA::MPI::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionAMG preconditioner_velocity;

    // Pressure mass matrix.
    const LA::MPI::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionAMG preconditioner_pressure;

    // B matrix.
    const LA::MPI::SparseMatrix *B;

    // D1 matrix.
    const LA::MPI::SparseMatrix *D1;

    // D2 matrix.
    const LA::MPI::SparseMatrix *D2;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionAMG preconditioner_solid;

    // Solid matrix.
    const LA::MPI::SparseMatrix *solid_matrix;

    // Temporary vector stokes
    mutable LA::MPI::Vector tmpStokes;
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
  ParameterHandler  &prm;

  // Number of MPI processes.
  // parallel::fullydistributed::Triangulation<dim> triangulation; doesn't
  // work
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

  SparsityPattern            sparsity_pattern;
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