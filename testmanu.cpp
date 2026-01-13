#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>
#include <iostream>
#include <vector>

using namespace dealii;

constexpr int dim = 2;

// -------------------- Soluzione esatta --------------------
template <int dim>
class ExactSolution : public Function<dim>
{
public:
    double value(const Point<dim> &p, unsigned int = 0) const override
    {
        double v = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
            v += p[d] * p[d];
        return v;
    }

    Tensor<1, dim> gradient(const Point<dim> &p, unsigned int = 0) const override
    {
        Tensor<1, dim> g;
        for (unsigned int d = 0; d < dim; ++d)
            g[d] = 2.0 * p[d];
        return g;
    }
};

// -------------------- Calcolo errore --------------------
double compute_blockvector_error(const DoFHandler<dim> &dof_handler,
                                 const std::vector<TrilinosWrappers::MPI::Vector> &blocks,
                                 const Function<dim> &exact_solution,
                                 const VectorTools::NormType &norm_type)
{
    FE_Q<dim> fe(1);
    QGauss<dim> quadrature(fe.degree + 2);

    Vector<double> error_per_cell(dof_handler.get_triangulation().n_active_cells());

    // Concateno i blocchi in un unico vettore locale
    unsigned int total_size = 0;
    for (auto &b : blocks)
        total_size += b.size();

    Vector<double> local_solution(total_size);
    unsigned int offset = 0;

    for (auto &b : blocks)
    {
        for (unsigned int i = 0; i < b.size(); ++i)
            local_solution[offset + i] = b[i];
        offset += b.size();
    }

    VectorTools::integrate_difference(dof_handler,
                                      local_solution,
                                      exact_solution,
                                      error_per_cell,
                                      quadrature,
                                      norm_type);

    return VectorTools::compute_global_error(dof_handler.get_triangulation(),
                                             error_per_cell,
                                             norm_type);
}

// -------------------- MAIN --------------------
int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(2);

    FE_Q<dim> fe(1);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    IndexSet locally_owned = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

    // -------------------- Blocchi MPI Vector --------------------
    std::vector<TrilinosWrappers::MPI::Vector> solution_blocks(2);
    for (auto &v : solution_blocks)
        v.reinit(locally_owned, locally_relevant, mpi_communicator),
        v = 0;

    ExactSolution<dim> exact_solution;

    double l2_error = compute_blockvector_error(dof_handler, solution_blocks, exact_solution, VectorTools::L2_norm);
    double h1_error = compute_blockvector_error(dof_handler, solution_blocks, exact_solution, VectorTools::H1_seminorm);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::cout << "L2 error = " << l2_error << std::endl;
        std::cout << "H1 error = " << h1_error << std::endl;
    }

    return 0;
}
