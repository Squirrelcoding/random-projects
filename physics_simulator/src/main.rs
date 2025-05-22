use std::f32::consts::PI;

use bevy::{
    math::ops::{cos, sin},
    prelude::*,
};

#[cfg(not(target_arch = "wasm32"))]
use rand::prelude::*;

const G: f32 = 6.67430e-11;

/// A vector representing a particle's velocity in the physics simulation.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct Velocity(Vec3);

/// A vector representing a particle's force in the physics simulation.
#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct Force(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct PhysicalTranslation(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct PreviousPhysicalTranslation(Vec3);

#[derive(Debug, Component, Clone, Copy, PartialEq, Default, Deref, DerefMut)]
struct Mass(f32);

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

        let theta = 2.0 * PI * rng.random::<f32>();
        let v_x = 210.0 * cos(theta);
        let v_y = 210.0 * sin(theta);

        let mass = 100.0 * rng.random::<f32>();

        let color = Color::linear_rgb(1.0, 0.0, 0.0);
        commands.spawn((
            Mesh2d(circle),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(x, y, 0.0),
            PhysicalTranslation(Vec3::new(x, y, 0.0)),
            PreviousPhysicalTranslation(Vec3::new(x, y, 0.0)),
            Velocity(Vec3::new(v_x, v_y, 0.0)),
            Force::default(),
            Mass(mass),
        ));
    }
}

fn advance_physics(
    fixed_time: Res<Time<Fixed>>,
    mut query: Query<(
        Entity,
        &mut PhysicalTranslation,
        &mut PreviousPhysicalTranslation,
        &mut Velocity,
        &mut Force,
        &Mass,
    )>,
    position_query: Query<(Entity, &PhysicalTranslation, &Mass)>,
) {
    // First collect all positions and masses into a temporary vector
    let positions_and_masses: Vec<_> = position_query
        .iter()
        .map(|(entity, pos, mass)| (entity, pos.0, mass.0))
        .collect();

    for (
        entity_a,
        mut current_physical_translation,
        mut previous_physical_translation,
        mut velocity,
        mut force,
        mass_a,
    ) in query.iter_mut()
    {
        let position_a = current_physical_translation.0;
        force.0 = Vec3::ZERO; // Reset force before accumulation
        
        // Calculate the force from all other bodies
        for (entity_b, position_b, mass_b) in &positions_and_masses {
            if entity_a == *entity_b {
                continue;
            }
            let distance = position_a.distance(*position_b);
            let g_force = (G * mass_a.0 * *mass_b) / distance;
            let force_vector = (*position_b - position_a).normalize() * g_force;

            force.0 += force_vector;
        }

        previous_physical_translation.0 = current_physical_translation.0;
        current_physical_translation.0 += velocity.0 * fixed_time.delta_secs();
        velocity.0 += force.0 / mass_a.0 * fixed_time.delta_secs(); // F=ma => a=F/m
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
