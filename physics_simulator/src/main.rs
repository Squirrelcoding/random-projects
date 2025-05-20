use bevy::prelude::*;
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

        let color = Color::linear_rgb(1.0, 0.0, 0.0);
        commands.spawn((
            Mesh2d(circle),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(x, y, 0.0),
            Velocity(Vec3 {
                x: 210.0,
                y: 210.0,
                z: 0.0,
            }),
            PhysicalTranslation::default(),
            PreviousPhysicalTranslation::default(),
        ));
    }
}

// #[cfg(not(target_arch = "wasm32"))]
// fn redraw_circles(
// mut commands: Commands,
// mut meshes: ResMut<Assets<Mesh>>,
// keyboard: Res<ButtonInput<KeyCode>>,
// mut materials: ResMut<Assets<ColorMaterial>>,
// ) {
//     if keyboard.just_pressed(KeyCode::Space) {

//     }
// }

fn advance_physics(
    fixed_time: Res<Time<Fixed>>,
    mut query: Query<(
        &mut PhysicalTranslation,
        &mut PreviousPhysicalTranslation,
        &mut AccumulatedInput,
        &Velocity,
    )>,
) {
    for (
        mut current_physical_translation,
        mut previous_physical_translation,
        mut input,
        velocity,
    ) in query.iter_mut()
    {
        previous_physical_translation.0 = current_physical_translation.0;
        current_physical_translation.0 += velocity.0 * fixed_time.delta_secs();

        // Reset the input accumulator, as we are currently consuming all input that happened since the last fixed timestep.
        input.0 = Vec2::ZERO;
        velocity.0;
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

fn handle_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut AccumulatedInput, &mut Velocity)>,
) {
    /// Since Bevy's default 2D camera setup is scaled such that
    /// one unit is one pixel, you can think of this as
    /// "How many pixels per second should the player move?"
    const SPEED: f32 = 210.0;
    for (mut input, mut velocity) in query.iter_mut() {
        if keyboard_input.pressed(KeyCode::KeyW) {
            input.y += 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyS) {
            input.y -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyA) {
            input.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            input.x += 1.0;
        }

        // Need to normalize and scale because otherwise
        // diagonal movement would be faster than horizontal or vertical movement.
        // This effectively averages the accumulated input.
        velocity.0 = input.extend(0.0).normalize_or_zero() * SPEED;
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, advance_physics)
        .add_systems(
            RunFixedMainLoop,
            (
                handle_input.in_set(RunFixedMainLoopSystem::BeforeFixedMainLoop),
                interpolate_rendered_transform.in_set(RunFixedMainLoopSystem::AfterFixedMainLoop),
            ),
        )
        .run();
}
