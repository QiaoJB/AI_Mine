from mcp.server.fastmcp import FastMCP
import threading
import uuid
import json

mcp = FastMCP("AI-Mine-Tools")

@mcp.tool()
def generate_lithology_features(image_path: str, mineral_type: str) -> str:
    """
    Generate lithological features from remote sensing imagery.

    This tool extracts lithology-related spectral features from multispectral
    satellite imagery (e.g., Sentinel-2). The generated features highlight
    spectral patterns associated with different rock types that are relevant
    to a specific mineral exploration target.

    Parameters
    ----------
    image_path : str
        Path to the multispectral satellite image directory or main image file.
        The image should contain required bands (e.g., Sentinel-2 bands).

    mineral_type : str
        Target mineral type for exploration (e.g., "gold", "copper", "iron", "coal").
        Different mineral types may use different lithological feature configurations.

    Returns
    -------
    str
        Path to the generated lithology feature file (e.g., lithology_features.npy).
    """
    from core.lithology import generate_lithology
    try:
        result = generate_lithology(image_path,mineral_type)
    except Exception as e:
        result = f"lithology_features.npy generate failed. str{e}"
    return result

@mcp.tool()
def generate_structure_features(image_path: str, dem_path: str) -> str:
    """
    Generate structural geological features from satellite imagery and DEM.

    This tool extracts structural information such as lineaments, faults,
    ridges, and terrain-derived features using satellite imagery and
    digital elevation models (DEM). Structural features are important
    geological controls for mineralization.

    Parameters
    ----------
    image_path : str
        Path to the multispectral satellite image directory or main image file.
        The image should contain required bands (e.g., Sentinel-2 bands).

    dem_path : str
        Path to the Digital Elevation Model (DEM) used for terrain analysis
        and structural feature extraction.
        The path must include the dem file with its suffix.

    Returns
    -------
    str
        Path to the generated structural feature file (e.g., structure_features.npy).
    """
    from core.structure import generate_structure
    try:
        result = generate_structure(image_path,dem_path)
    except Exception as e:
        result = f"structure_features.npy generate failed. str{e}"
    return result

@mcp.tool()
def generate_alteration_features(image_path: str, mineral_type: str) -> str:
    """
    Generate hydrothermal alteration features from multispectral imagery.

    This tool calculates a set of spectral alteration indices that highlight
    hydrothermal alteration minerals commonly associated with mineral deposits.
    Examples include iron oxide, hydroxyl-bearing minerals, silica, and clay.

    Environmental suppression factors such as vegetation (NDVI) and moisture
    may also be applied to reduce spectral interference.

    Parameters
    ----------
    image_path : str
        Path to the multispectral satellite image directory or main image file.
        The image should contain required bands (e.g., Sentinel-2 bands).

    mineral_type : str
        Target mineral type (e.g., "gold", "copper", "iron", "coal").
        Different minerals may use different alteration indices.

    Returns
    -------
    str
        Path to the generated alteration feature stack (e.g., alter_stack.npy).
    """
    from core.alteration import generate_alteration
    try:
        result = generate_alteration(image_path,mineral_type)
    except Exception as e:
        result = f"alter_stack.npy generate failed. str{e}"
    return result

@mcp.tool()
def build_feature_cube(image_path: str) -> str:
    """
    Build a unified feature cube for mineral prospectivity analysis.

    This tool integrates lithological features, structural features,
    and hydrothermal alteration features into a multi-dimensional
    feature cube. Each pixel in the cube contains a set of geological
    features used for machine learning-based mineral prediction.

    The feature cube serves as the main input for subsequent
    mineral prospectivity modeling.

    Parameters
    ----------
    image_path : str
        Path to the input satellite image or project directory.

    Returns
    -------
    str
        Path to the generated feature cube file (e.g., feature_cube.npy).
    """
    from core.feature_cube import build_feature_cube
    try:
        result = build_feature_cube(image_path)
    except Exception as e:
        result = f"feature_cube.npy generate failed. str{e}"
    return result


tasks = {}
def run_prediction(task_id, image_path):
    from core.prediction import generate_prediction
    try:
        result = generate_prediction(image_path)

        tasks[task_id]["status"] = "finished"
        tasks[task_id]["result"] = result
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["result"] = str(e)
@mcp.tool()
def submit_prediction(image_path: str) -> str:
    """
    Submit a mineral prospectivity prediction task.

    This tool starts an asynchronous mineral prediction process based on
    previously generated geological features. Because prediction over large
    remote sensing datasets may take a long time, the task runs in the
    background and immediately returns a task ID.

    The returned task ID can be used to query the prediction status and
    retrieve the final result using the `get_prediction_result` tool.

    Typical workflow:
        1. Call submit_prediction to start the prediction task
        2. Receive a task_id
        3. Periodically call get_prediction_result(task_id) to check progress
        4. When status becomes "finished", retrieve the prediction output

    Parameters
    ----------
    image_path : str
        Path to the project directory containing the feature cube and
        required geological feature data.

    Returns
    -------
    dict
        A dictionary containing:

        task_id : str
            Unique identifier for the submitted prediction task.

        status : str
            Current task status. Initially "running".
    """
    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status": "running",
        "result": None
    }

    thread = threading.Thread(
        target=run_prediction,
        args=(task_id, image_path)
    )
    thread.start()

    return json.dumps({
        "task_id": task_id,
        "status": "running"
    })
    

@mcp.tool()
def get_prediction_result(task_id: str) -> str:
    """
    Retrieve the status or result of a submitted mineral prediction task.

    This tool checks the execution status of a previously submitted
    prediction task. The task must first be created using
    `submit_prediction`.

    Possible task states:

        running   : prediction is still processing
        finished  : prediction completed successfully
        failed    : prediction failed (if error handling is implemented)

    When the task is finished, the result typically contains the path
    to the generated mineral prospectivity map.

    Parameters
    ----------
    task_id : str
        Unique task identifier returned by `submit_prediction`.

    Returns
    -------
    dict
        A dictionary containing:

        status : str
            Current status of the task ("running" or "finished" or "failed").

        result : str
            Path to the generated prediction result
            when the task is finished. Otherwise may be None.
    """
    if task_id not in tasks:
        return {"error": "task not found"}
    
    return json.dumps(tasks[task_id])

@mcp.tool()
def mineral_exploration_workflow() -> dict:
    """
    Describe the mineral exploration workflow and tool dependencies.
    """

    return {
        "parallel_steps": [
            "generate_lithology_features",
            "generate_structure_features",
            "generate_alteration_features"
        ],
        "next_step": "build_feature_cube",
        "final_step": [
            "submit_prediction",
            "get_prediction_result"
        ]
    }

@mcp.tool()
def view_raster_tiff(tiff_path: str) -> str:
    """
    Visualize a GeoTIFF/TIFF raster file.

    This tool reads a raster dataset and displays basic information
    and a preview visualization.

    Supported features:
    - Single-band raster visualization
    - Multi-band RGB visualization
    - Automatic normalization for display

    Parameters
    ----------
    tiff_path : str
        Path to the TIFF/GeoTIFF file.

    Returns
    -------
    str
        Summary of raster information including size, band count,
        CRS, and data type.
    """
    from core.view_raster_tiff import view_raster_tiff
    try:
        result = view_raster_tiff(tiff_path)
    except Exception as e:
        result = f"Raster loaded failed. str{e}"
    return result

@mcp.tool()
def view_npy_features(npy_path: str) -> str:
    """
    Visualize numpy feature arrays.

    Supports:
    - [C,H,W] feature stacks
    - [N_pixels,C] flattened features

    Parameters
    ----------
    npy_path : str
        Path to the .npy feature file.

    Returns
    -------
    str
        Feature information summary.
    """
    from core.view_npy_features import view_npy_features
    try:
        result = view_npy_features(npy_path)
    except Exception as e:
        result = f"npy loaded failed. str{e}"
    return result

if __name__ == "__main__":
    print("MCP server starting...")
    mcp.run()