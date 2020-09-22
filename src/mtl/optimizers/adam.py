import transformers

"""
    check out here for adding more optimizers and schedulers:
    https://huggingface.co/transformers/main_classes/optimizer_schedules.html
"""

__all__ = ["AdamW"]

def AdamW():
    """
    Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay Regularization.
    """
    return transformers.AdamW