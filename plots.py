import graphviz


def plot_dgp_dag():
    """Generate directed acyclic graph describing data generating process."""
    dot = graphviz.Digraph("dgp", comment="Data Generating Process", format="png")
    # TODO: To make a node/edge invisible use `style="invis"`.
    dot.node("Age", "Age")
    dot.node("Nationality", "Nationality")
    dot.node("Chef_rating", "Chef rating")
    dot.node("Gaz_stove", "Gaz stove")
    dot.node("Stirring", "Stirring")
    dot.node("Pleasure", "Pleasure")
    dot.edge("Chef_rating", "Gaz_stove")
    for covariate in ["Age", "Nationality", "Chef_rating", "Gaz_stove"]:
        dot.edge(covariate, "Pleasure")
        dot.edge(covariate, "Stirring")
    dot.edge("Stirring", "Pleasure")
    dot.render(directory="plots").replace("\\", "/")
    return dot


plot_dgp_dag()
