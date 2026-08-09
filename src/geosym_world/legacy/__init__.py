"""Complete compatibility surface for the original ``legacy_tools`` package.

The aliases intentionally point to the original implementations. This preserves
their full public and private method surface while the optimized orchestrator in
``legacy.pipeline`` provides deterministic, parameterized execution.
"""

from geosym_world.legacy_tools import builder, drawer, gt, qa_template, region, shader, shaders, template
from geosym_world.legacy_tools.builder import RandomGeometryBuilder
from geosym_world.legacy_tools.drawer import GeometryDrawer
from geosym_world.legacy_tools.gt import GeometryCalculator
from geosym_world.legacy_tools.qa_template import QAGenerator
from geosym_world.legacy_tools.region import RegionExtractConfig, RegionExtractor
from geosym_world.legacy_tools.shader import CoordTransformer, EnhancedDrawer, GeometryAnnotator
from geosym_world.legacy_tools.shaders import CrossHatchShader, GradientShader, HatchShader, SHADERS, Shader, SolidShader
from geosym_world.legacy_tools.template import TemplateGenerator
from .pipeline import LegacyCompatiblePipeline, LegacyPipelineOptions
from .registry import LEGACY_CAPABILITIES, compatibility_report

from geosym_world.legacy_tools import utils

extract_answer = utils.extract_answer
extract_boxed_answer = utils.extract_boxed_answer
grade_answer_mathd = utils.grade_answer_mathd
grade_answer_sympy = utils.grade_answer_sympy
grade_answer_verl = utils.grade_answer_verl
mathd_normalize_answer = utils.mathd_normalize_answer

__all__ = [
    "CoordTransformer",
    "CrossHatchShader",
    "EnhancedDrawer",
    "GeometryAnnotator",
    "GeometryCalculator",
    "GeometryDrawer",
    "GradientShader",
    "HatchShader",
    "LEGACY_CAPABILITIES",
    "LegacyCompatiblePipeline",
    "LegacyPipelineOptions",
    "QAGenerator",
    "RandomGeometryBuilder",
    "RegionExtractConfig",
    "RegionExtractor",
    "SHADERS",
    "Shader",
    "SolidShader",
    "TemplateGenerator",
    "builder",
    "compatibility_report",
    "drawer",
    "extract_answer",
    "extract_boxed_answer",
    "grade_answer_mathd",
    "grade_answer_sympy",
    "grade_answer_verl",
    "gt",
    "mathd_normalize_answer",
    "qa_template",
    "region",
    "shader",
    "shaders",
    "template",
    "utils",
]
