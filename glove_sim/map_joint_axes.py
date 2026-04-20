import json
import numpy as np
import trimesh
import pyglet
from pathlib import Path
from urdfpy import URDF
from trimesh.viewer.windowed import SceneViewer

class CalibrationViewer(SceneViewer):
    """
    Custom Trimesh viewer that captures keystrokes without closing the window,
    updating the joint poses in real-time.
    """
    def __init__(self, scene, robot, moving_joints, posed_nodes, output_path, **kwargs):
        self._scene = scene
        self.robot = robot
        self.moving_joints = moving_joints
        self.posed_nodes = posed_nodes
        self.output_path = output_path
        
        self.current_idx = 0
        self.results = {}
        self.test_angle = 0.8
        
        # Initialize the first pose before drawing the window
        self.update_pose() 
        
        # Start the Pyglet window loop
        super().__init__(scene, **kwargs)

    def update_pose(self):
        if self.current_idx >= len(self.moving_joints):
            return
        
        joint = self.moving_joints[self.current_idx]
        print(f"\n--- Testing: {joint.name} ---")
        print("Press 'T' (towards), 'A' (away), or 'N' (sideways/NA).")
        
        # Calculate new kinematics for just this joint
        cfg = {joint.name: self.test_angle}
        posed_fk = self.robot.visual_trimesh_fk(cfg=cfg)
        
        # Instantly snap the active meshes to their new solved positions
        scene = getattr(self, "scene", self._scene)
        for base_mesh, node_name in self.posed_nodes:
            if base_mesh in posed_fk:
                new_transform = posed_fk[base_mesh]
                scene.graph.update(frame_to=node_name, matrix=new_transform)

    def on_key_press(self, symbol, modifiers):
        # Let trimesh handle camera controls (like W/A/S/D) if we are done
        if self.current_idx >= len(self.moving_joints):
            super().on_key_press(symbol, modifiers)
            return

        joint = self.moving_joints[self.current_idx]
        
        # Map keystrokes
        if symbol == pyglet.window.key.T:
            self.results[joint.name] = "towards"
        elif symbol == pyglet.window.key.A:
            self.results[joint.name] = "away"
        elif symbol == pyglet.window.key.N:
            self.results[joint.name] = "na"
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()
            return
        else:
            # Let standard camera keys pass through
            super().on_key_press(symbol, modifiers)
            return
            
        print(f"✅ Recorded {joint.name} -> {self.results[joint.name]}")
        self.current_idx += 1
        
        if self.current_idx < len(self.moving_joints):
            self.update_pose() # Animate to the next joint
        else:
            print("\nAll joints tested! Saving results...")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"Saved to {self.output_path}")
            self.close() # Close window automatically when done

def main():
    # 1. Setup paths
    root = Path(r"C:/Users/dnxjc/Desktop/CURR/PROJECTS/rewind")
    urdf_path = root / "rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf"
    output_path = root / "glove_sim/outputs/aligned/joint_directions.json"
    
    print(f"Loading URDF from: {urdf_path}")
    robot = URDF.load(str(urdf_path))
    
    moving_joints = [j for j in robot.joints if j.joint_type in ['revolute', 'continuous']]
    
    # 2. Build the Scene
    scene = trimesh.Scene()
    rest_fk = robot.visual_trimesh_fk(cfg={})
    
    posed_nodes = []
    
    for i, (mesh, transform) in enumerate(rest_fk.items()):
        # --- Ghost Layer (Static) ---
        rest_mesh = mesh.copy()
        rest_mesh.visual.face_colors = [100, 150, 255, 80] # Translucent Blue
        scene.add_geometry(rest_mesh, transform=transform, node_name=f"rest_node_{i}")
        
        # --- Active Layer (Dynamic) ---
        posed_mesh = mesh.copy()
        posed_mesh.visual.face_colors = [200, 200, 200, 255] # Solid Grey for contrast
        
        # Add to scene, but save the exact node_name so we can move it later
        node_name = f"posed_node_{i}"
        scene.add_geometry(posed_mesh, transform=transform, node_name=node_name)
        posed_nodes.append((mesh, node_name))

    print("\n=== Interactive GUI Calibration ===")
    print("Keep the 3D window focused.")
    print("Use the mouse to rotate the camera.")
    
    # 3. Launch the custom viewer (blocks until finished)
    CalibrationViewer(
        scene=scene, 
        robot=robot, 
        moving_joints=moving_joints, 
        posed_nodes=posed_nodes, 
        output_path=output_path,
        resolution=(1280, 720) # Open a nice wide window
    )

if __name__ == "__main__":
    main()