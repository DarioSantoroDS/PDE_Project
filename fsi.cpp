#include <fstream>
#include <iostream>
#include <vector>

#include "FluidStructureProblem.hpp"

int
main(int argc, char *argv[])

{
  // try
  // {


#ifdef DEBUG
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  std::cout << "im in debug mode" << std::endl;
#else
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
#endif
  ParameterHandler prm;
  ParameterReader  param(prm);
  param.read_parameters("config.prm");
  prm.enter_subsection("Refinement");
  const unsigned int    n_refinement = prm.get_integer("Refinement cycles");
  prm.leave_subsection();
  FluidStructureProblem flow_problem(1, 1, prm);
  flow_problem.make_grid();
  for (unsigned int refinement_cycle = 0; refinement_cycle < n_refinement;
       ++refinement_cycle)
    {
      if (refinement_cycle > 0)
        flow_problem.refine_mesh();
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
      flow_problem.output_results(refinement_cycle);
    }
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

  double l2_error = flow_problem.compute_velocity_error(VectorTools::L2_norm);
  //double h1_error = flow_problem.compute_velocity_error(VectorTools::H1_seminorm);

  std::cout << "L2 velocity error: " << l2_error << std::endl;
  // std::cout << "H1 velocity error: " << h1_error << std::endl;

  return 0;
}