# models/__init__.py

from models.reflex_module import ReflexModule
from models.instinct_module import InstinctModule
from models.gan import GAN, PatternEvaluator, PatternRepository
from models.instinct_gan import InstinctGAN

__all__ = [
    'ReflexModule',
    'InstinctModule',
    'GAN',
    'PatternEvaluator',
    'PatternRepository',
    'InstinctGAN',
]



# # models/__init__.py
# from models.reflex_module import ReflexModule
# from models.instinct_module import InstinctModule
# from models.gan import GAN, ExperienceEncoder, PatternEvaluator, PatternRepository
#
# __all__ = [
#     'ReflexModule',
#     'InstinctModule',
#     'GAN',
#     'ExperienceEncoder',
#     'PatternEvaluator',
#     'PatternRepository'
# ]