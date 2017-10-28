"""
Setting methods that do not directly call the Netica C Library.

None of these methods are called by methods in netica.py.
"""


def setnodeCPT(self, node_p=None, df=None):
    """High level setting of node CPT using a Pandas DataFrame."""
    node_p = self.getnodenamed(node_p)  # Verify pointer.
    # Checks
    # TODO: Check if DataFrame matches that of getter.
    getdf = self.getnodedataframe(node_p)
    getdf *= 0
    df2 = df.copy()
    df2 *= 0

    if getdf.equals(df2):
        print("Warning: The indices of the input DataFrame " +
              "don't match those of the CPT.")

    arr = df.values
    self.setnodeprobs(node_p, probs=arr)
