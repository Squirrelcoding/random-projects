//! Shows how to render simple primitive shapes with a single color.
//!
//! You can toggle wireframes with the space bar except on wasm. Wasm does not support
//! `POLYGON_MODE_LINE` on the gpu.

use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::sprite::Wireframe2dPlugin;
use rand::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        #[cfg(not(target_arch = "wasm32"))]
        Wireframe2dPlugin::default(),
    ))
    .add_systems(Startup, setup);
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, redraw_circles);
    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

const X_EXTENT: f32 = 900.;

#[cfg(not(target_arch = "wasm32"))]
fn redraw_circles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let circle = meshes.add(Circle::new(20.0));
            // Generate random X and Y coordinate, map it to [-1, 1], and multiply by the X_EXTEND.
            let x = (2.0 * rng.random::<f32>() - 1.0) * X_EXTENT;
            let y = (2.0 * rng.random::<f32>() - 1.0) * X_EXTENT;

            let color = Color::linear_rgb(1.0, 0.0, 0.0);
            commands.spawn((
                Mesh2d(circle),
                MeshMaterial2d(materials.add(color)),
                Transform::from_xyz(x, y, 0.0),
            ));
        }
    }
}
