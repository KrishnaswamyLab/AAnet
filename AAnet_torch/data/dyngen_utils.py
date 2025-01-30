import scprep
import pandas as pd

_install = scprep.run.RFunction(
        args="""lib=.libPaths()[1], dependencies=NA, verbose=TRUE""",
        body="""
        cat ('Installing package in:', lib, '\n')
        
        quiet <- !verbose

        if (!require('remotes')) install.packages('remotes')
        remotes::install_github('dynverse/dyngen',
                         repos = "http://cran.us.r-project.org",
                         lib=lib, dependencies=dependencies,
                         quiet=quiet)
        """)

_get_backbones = scprep.run.RFunction(
    args="""lib=.libPaths()[1]""",
    body="""
       library(dyngen, lib.loc=lib)
       names(list_backbones())
       """)

_generate_dataset = scprep.run.RFunction(
    args="""
    backbone_name=character(), num_cells=500, num_tfs=100, num_targets=50, num_hks=25,
    simulation_census_interval=10, n_jobs=8, lib=.libPaths()[1]
    """,
    body="""
    library(dyngen, lib.loc=lib)
    if (!(backbone_name %in% names(list_backbones()))) {
        stop("Input not in list of dyngen backbones. Choose name from get_backbones().")
    }
    backbone <- list_backbones()[[backbone_name]]()
    cat('Run Parameters:\n\tBackbone: ', backbone_name)
    cat('\n\tNumber of Cells: ', num_cells)
    cat('\n\tNumber of TFs: ', num_tfs)
    cat('\n\tNumber of Targets: ', num_targets)
    cat('\n\tNumber of HKs: ', num_hks, '\n')

    init <- initialise_model(
      backbone = backbone,
      num_cells = num_cells,
      num_tfs = num_tfs,
      num_targets = num_targets,
      num_hks = num_hks,
      simulation_params = simulation_default(census_interval = 10, 
                                      kinetics_noise_function=kinetics_noise_simple(mean = 1, sd = 0.005),
                                      ssa_algorithm = ssa_etl(tau = 300 / 3600)),
      num_cores = n_jobs
    )
    out <- generate_dataset(init)
    list(cell_info = as.data.frame(out$dataset$cell_info),
             expression = as.data.frame(as.matrix(out$dataset$expression)))
    """,
    verbose=False
)

def install(lib_loc=None, dependencies=None, verbose=True):
    """Install dyngen package and dependencies.
    
    Parameters
    ----------    
    lib_loc: string, optional (default: None)
             library directory in which to install the package.
             If None, defaults to the first element of .libPaths()
    dependencies: boolean, optional (default: None/NA)
             When True, installs all packages specified under "Depends", "Imports", "LinkingTo" and "Suggests".
             When False, installs no dependencies.
             When None/NA, installs all packages specified under "Depends", "Imports" and "LinkingTo".
    verbose: boolean, optional (default: True)
             Install script verbosity.
    """
    kwargs = {"verbose": verbose}
    if lib_loc is not None:
        kwargs["lib"] = lib_loc
    if dependencies is not None:
        kwargs["dependencies"] = dependencies
        
    _install(**kwargs)
    
    
def get_backbones(lib_loc=None):
    """Output full list of cell trajectory backbones.
    
    Parameters
    ----------
    lib_loc: string, optional (default: None)
             Library directory from which to load dyngen.
             If None, defaults to the first element of .libPaths()
             
    Returns
    -------
    backbones: array of backbone names
    """
    kwargs = {}
    if lib_loc is not None:
        kwargs["lib"] = lib_loc
    
    return(_get_backbones(**kwargs))

def generate_dataset(backbone, num_cells=500, num_tfs=100, num_targets=50, num_hks=25,
    simulation_census_interval=10, n_jobs=8, lib_loc=None):
    """Simulate dataset with backbone determining cellular dynamics. The backbone of a dyngen model is what determines the overall dynamic process that a cell will undergo during a simulation. It consists of a set of gene modules, which regulate eachother in such a way that expression of certain genes change over time in a specific manner.
    
    Parameters
    ----------
    backbone: string
           Backbone name from dyngen list of backbones.
           Get list with get_backbones()).
    num_cells: int, optional (default: 500)
           Number of cells.
    num_tfs: int, optional (default: 100)
           Number of transcription factors. The TFs are the main drivers of the molecular changes in the simulation. A TF can only be regulated by other TFs or itself.
    num_targets: int, optional (default: 50)
           Number of target genes. Target genes are regulated by a TF or another target gene, but are always downstream of at least one TF. 
    num_hks: int, optional (default: 25)
           Number of housekeeping genees. Housekeeping genes are completely separate from any TFs or target genes.
    simulation_census_interval: int, optional (default: 10)
           Allows storing the abundance levels of molecules only after a specific interval has passed since the previous census. By setting the census interval to 0, the whole simulation’s trajectory is retained but many of these time points will contain very similar information.
    n_jobs: int, optional (default: 8)
           Number of cores to use.
    lib_loc: string, optional (default: None)
             Library directory from which to load dyngen.
             If None, defaults to the first element of .libPaths()
             
    Returns
    -------
    data_cell_info: pd.DataFrame with columns: cell_id, step_ix, simulation_i, sim_time, num_molecules, mult, lib_size; sim_time is the simulated timepoint for a given cell.
    data_expression: pd.DataFrame with log-transformed counts with dropout for num_cells and total number of genes.
    """
    kwargs = {}
    if lib_loc is not None:
        kwargs["lib"] = lib_loc
       
    data = _generate_dataset(backbone_name=backbone,
                        num_cells=num_cells,
                        num_tfs=num_tfs,
                        num_targets=num_targets,
                        num_hks=num_hks,
                        simulation_census_interval=simulation_census_interval,
                        n_jobs=n_jobs,
                        **kwargs)
    
    data_cell_info = pd.DataFrame(data['cell_info'])
    data_expression = pd.DataFrame(data['expression'])
    
    return(data_cell_info, data_expression)