# from .aliev_panfilov import AlievPanfilov
# from .barkley import Barkley
# from .mitchell_schaeffer import MitchellSchaeffer
# from .fenton_karma import FentonKarma
# from .bueno_orovio import BuenoOrovio
# from .luo_rudy_91 import LuoRudy91
# from .ten_tusscher_panfilov_2006 import TenTusscherPanfilov2006
# from .courtemanche import Courtemanche
from .cardiac_model import CardiacModel

class AlievPanfilov(CardiacModel):
    model_name = "aliev_panfilov"

class Barkley(CardiacModel):
    model_name = "barkley"

class BuenoOrovio(CardiacModel):
    model_name = "bueno_orovio"

class Courtemanche(CardiacModel):
    model_name = "courtemanche"

class FentonKarma(CardiacModel):
    model_name = "fenton_karma"

class LuoRudy91(CardiacModel):
    model_name = "luo_rudy_91"

class MitchellSchaeffer(CardiacModel):
    model_name = "mitchell_schaeffer"

class TenTusscherPanfilov2006(CardiacModel):
    model_name = "ten_tusscher_panfilov_2006"