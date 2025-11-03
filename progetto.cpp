#include <fstream>
#include <iostream>
#include <vector>

#include "FluidStructureProblem.hpp"

int main(int argc, char *argv[])

{
    try
    {
        FluidStructureProblem flow_problem(1, 1);
        flow_problem.make_grid();

        for (unsigned int refinement_cycle = 0; refinement_cycle < 10 - 2 * flow_problem.dim;
             ++refinement_cycle)
        {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;

            if (refinement_cycle > 0)
                flow_problem.refine_mesh();

            flow_problem.setup_dofs();

            std::cout << "   Assembling..." << std::endl;
            flow_problem.assemble_system();

            std::cout << "   Solving..." << std::endl;
            flow_problem.solve();

            std::cout << "   Writing output..." << std::endl;
            flow_problem.output_results(refinement_cycle);

            std::cout << std::endl;
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}