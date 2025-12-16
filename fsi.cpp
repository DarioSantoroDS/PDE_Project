#include <fstream>
#include <iostream>
#include <vector>

#include "FluidStructureProblem.hpp"

int main(int argc, char* argv[])

{
    // try
    // {


#ifdef DEBUG
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);
    std::cout << "im in debug mode" << std::endl;
#else
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
#endif
    ParameterHandler prm;
    ParameterReader  param(prm);
    param.read_parameters("config.prm");
    FluidStructureProblem flow_problem(1, 1,prm);
        flow_problem.make_grid();
        flow_problem.setup_dofs();

        flow_problem.pcout << "   Assembling..." << std::endl;
        flow_problem.assemble_system();
#ifdef DEBUG
        flow_problem.output_matrix();
#endif
        flow_problem.pcout << "   Solving..." << std::endl;
        flow_problem.solve_iterative();
        // flow_problem.solve();

        flow_problem.pcout << "   Writing output..." << std::endl;
        flow_problem.output_results(0);

        flow_problem.pcout << std::endl;
    // }

    // catch (std::exception& exc)
    // {
    //     std::cerr << std::endl
    //         << std::endl
    //         << "----------------------------------------------------"
    //         << std::endl;
    //     std::cerr << "Exception on processing: " << std::endl
    //         << exc.what() << std::endl
    //         << "Aborting!" << std::endl
    //         << "----------------------------------------------------"
    //         << std::endl;

    //     return 1;
    // }
    // catch (...)
    // {
    //     std::cerr << std::endl
    //         << std::endl
    //         << "----------------------------------------------------"
    //         << std::endl;
    //     std::cerr << "Unknown exception!" << std::endl
    //         << "Aborting!" << std::endl
    //         << "----------------------------------------------------"
    //         << std::endl;
    //     return 1;
    // }

    return 0;
}