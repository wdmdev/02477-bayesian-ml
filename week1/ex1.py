from daft import PGM

def show_beta_binomial_pgm():
    pgm = PGM()

    # Hyperparameters
    pgm.add_node("a", r"$a_0$", 0, 2, fixed=True)
    pgm.add_node("b", r"$b_0$", 1, 2, fixed=True)

    # Latent variable 
    pgm.add_node("mu", r"$\mu$", 0.5, 1.5)

    # Data
    pgm.add_node("Y", r"$Y$", 0.5, 0.5, observed=True)

    # Add in the edges.
    pgm.add_edge("a", "mu")
    pgm.add_edge("b", "mu")
    pgm.add_edge("mu", "Y")

 
    # Render and save.
    ax = pgm.render(dpi=100)