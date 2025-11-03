#ifndef FluidStructureInteractionProblem
#define FluidStructureInteractionProblem

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>

using namespace dealii;

class FluidStructureProblem
{
public:
    static constexpr unsigned int dim = 2;

    FluidStructureProblem(
        const unsigned int stokes_degree,
        const unsigned int elasticity_degree)
        : stokes_degree(stokes_degree), elasticity_degree(elasticity_degree), triangulation(Triangulation<dim>::maximum_smoothing), stokes_fe(FE_Q<dim>(stokes_degree + 1),
                                                                                                                                              dim,
                                                                                                                                              FE_Q<dim>(stokes_degree),
                                                                                                                                              1,
                                                                                                                                              FE_Nothing<dim>(),
                                                                                                                                              dim),
          elasticity_fe(FE_Nothing<dim>(),
                        dim,
                        FE_Nothing<dim>(),
                        1,
                        FE_Q<dim>(elasticity_degree),
                        dim),
          dof_handler(triangulation), viscosity(2), lambda(1), mu(1)
    {
        fe_collection.push_back(stokes_fe);
        fe_collection.push_back(elasticity_fe);
    }
    void make_grid();
    void setup_dofs();
    void assemble_system();
    void solve();
    void output_results(const unsigned int refinement_cycle) const;
    void refine_mesh();

    class StokesBoundaryValues : public Function<dim>
    {
    public:
        StokesBoundaryValues()
            : Function<dim>(dim + 1 + dim)
        {
        }

        virtual double value(const Point<dim> &p,
                             const unsigned int component = 0) const override
        {
            Assert(component < this->n_components,
                   ExcIndexRange(component, 0, this->n_components));

            if (component == dim - 1)
                switch (dim)
                {
                case 2:
                    return std::sin(numbers::PI * p[0]);
                case 3:
                    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
                default:
                    Assert(false, ExcNotImplemented());
                }

            return 0;
        }

        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const override
        {
            for (unsigned int c = 0; c < this->n_components; ++c)
                values(c) = StokesBoundaryValues::value(p, c);
        }
    };

private:
    enum
    {
        fluid_domain_id,
        solid_domain_id
    };

    static bool cell_is_in_fluid_domain(
        const typename DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == fluid_domain_id);
    }

    static bool cell_is_in_solid_domain(
        const typename DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == solid_domain_id);
    }

    void set_active_fe_indices();
    void assemble_interface_term(
        const FEFaceValuesBase<dim> &elasticity_fe_face_values,
        const FEFaceValuesBase<dim> &stokes_fe_face_values,
        std::vector<Tensor<1, dim>> &elasticity_phi,
        std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
        std::vector<double> &stokes_phi_p,
        FullMatrix<double> &local_interface_matrix) const;

    const unsigned int stokes_degree;
    const unsigned int elasticity_degree;

    Triangulation<dim> triangulation;
    FESystem<dim> stokes_fe;
    FESystem<dim> elasticity_fe;
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    const double viscosity;
    const double lambda;
    const double mu;
};

#endif