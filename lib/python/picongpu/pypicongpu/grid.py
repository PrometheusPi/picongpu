from . import util
from typeguard import typechecked
import enum
from .rendering import RenderedObject


@typechecked
class BoundaryCondition(enum.Enum):
    """
    Boundary Condition of PIConGPU

    Defines how particles that pass the simulation bounding box are treated.

    TODO: implement the other methods supported by PIConGPU
    (reflecting, thermal)
    """
    PERIODIC = 1
    ABSORBING = 2

    def get_cfg_str(self) -> str:
        """
        Get string equivalent for cfg files
        :return: string for --periodic
        """
        literal_by_boundarycondition = {
            BoundaryCondition.PERIODIC: "1",
            BoundaryCondition.ABSORBING: "0",
        }
        return literal_by_boundarycondition[self]


@typechecked
class Grid3D(RenderedObject):
    """
    PIConGPU 3 dimensional (cartesian) grid

    Defined by the dimensions of each cell and the number of cells per axis.

    The bounding box is implicitly given as TODO.
    """

    cell_size_x_si = util.build_typesafe_property(float)
    """Width of individual cell in X direction"""
    cell_size_y_si = util.build_typesafe_property(float)
    """Width of individual cell in Y direction"""
    cell_size_z_si = util.build_typesafe_property(float)
    """Width of individual cell in Z direction"""

    cell_cnt_x = util.build_typesafe_property(int)
    """total number of cells in X direction"""
    cell_cnt_y = util.build_typesafe_property(int)
    """total number of cells in Y direction"""
    cell_cnt_z = util.build_typesafe_property(int)
    """total number of cells in Z direction"""

    boundary_condition_x = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the X boundary"""
    boundary_condition_y = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the Y boundary"""
    boundary_condition_z = util.build_typesafe_property(BoundaryCondition)
    """behavior towards particles crossing the Z boundary"""

    def _get_serialized(self) -> dict:
        """serialized representation provided for RenderedObject"""
        return {
            "cell_size": {
                "x": self.cell_size_x_si,
                "y": self.cell_size_y_si,
                "z": self.cell_size_z_si,
            },
            "cell_cnt": {
                "x": self.cell_cnt_x,
                "y": self.cell_cnt_y,
                "z": self.cell_cnt_z,
            },
            "boundary_condition": {
                "x": self.boundary_condition_x.get_cfg_str(),
                "y": self.boundary_condition_y.get_cfg_str(),
                "z": self.boundary_condition_z.get_cfg_str(),
            }
        }
