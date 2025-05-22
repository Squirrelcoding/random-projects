use std::f32::consts::PI;

use bevy::{math::ops::{cos, sin, sqrt}, prelude::*};
#[cfg(not(target_arch = "wasm32"))]
use rand::prelude::*;

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct AccumulatedInput(Vec2);

/// A vector representing a ball's velocity in the physics simulation.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct Velocity(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct PhysicalTranslation(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct PreviousPhysicalTranslation(Vec3);

const X_EXTENT: f32 = 900.;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);
    let mut rng = rand::rng();
    for _ in 0..100 {
        let circle = meshes.add(Circle::new(20.0));
        // Generate random X and Y coordinate, map it to [-1, 1], and multiply by the X_EXTEND.
        let x = (2.0 * rng.random::<f32>() - 1.0) * X_EXTENT;
        let y = (2.0 * rng.random::<f32>() - 1.0) * X_EXTENT;
        println!("Setting circle at ({x}, {y}).");

        let theta = 2.0 * PI * rng.random::<f32>();
        let v_x = 210.0 * cos(theta);
        let v_y = 210.0 * sin(theta);
        
        let color = Color::linear_rgb(1.0, 0.0, 0.0);
        commands.spawn((
            Mesh2d(circle),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(x, y, 0.0),
            PhysicalTranslation(Vec3::new(x, y, 0.0)),
            PreviousPhysicalTranslation(Vec3::new(x, y, 0.0)),
            Velocity(Vec3::new(v_x, v_y, 0.0)),
        ));
    }
}

fn advance_physics(
    fixed_time: Res<Time<Fixed>>,
    mut query: Query<(
        &mut PhysicalTranslation,
        &mut PreviousPhysicalTranslation,
        &Velocity,
    )>,
) {
    for (mut current_physical_translation, mut previous_physical_translation, velocity) in
        query.iter_mut()
    {
        previous_physical_translation.0 = current_physical_translation.0;
        current_physical_translation.0 += velocity.0 * fixed_time.delta_secs();
        // Reset the input accumulator, as we are currently consuming all input that happened since the last fixed timestep.
    }
}

fn interpolate_rendered_transform(
    fixed_time: Res<Time<Fixed>>,
    mut query: Query<(
        &mut Transform,
        &PhysicalTranslation,
        &PreviousPhysicalTranslation,
    )>,
) {
    for (mut transform, current_physical_translation, previous_physical_translation) in
        query.iter_mut()
    {
        let previous = previous_physical_translation.0;
        let current = current_physical_translation.0;
        // The overstep fraction is a value between 0 and 1 that tells us how far we are between two fixed timesteps.
        let alpha = fixed_time.overstep_fraction();

        let rendered_translation = previous.lerp(current, alpha);
        transform.translation = rendered_translation;
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, advance_physics)
        .add_systems(
            RunFixedMainLoop,
            (interpolate_rendered_transform.in_set(RunFixedMainLoopSystem::AfterFixedMainLoop),),
        )
        .run();
}
