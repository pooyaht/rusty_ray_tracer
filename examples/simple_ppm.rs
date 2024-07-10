use rusty_ray_tracer as raytracer;

struct Particle {
    position: raytracer::vector::Point,
    velocity: raytracer::vector::Vector3,
}

struct World {
    gravity: raytracer::vector::Vector3,
    wind: raytracer::vector::Vector3,
}

fn main() {
    let mut canvas = raytracer::canvas::Canvas::new(100, 100);
    let world = World {
        gravity: raytracer::vector::Vector3::new(0.0, -0.01, 0.0),
        wind: raytracer::vector::Vector3::new(-0.001, 0.0, 0.0),
    };
    let mut particle = Particle {
        position: raytracer::vector::Point::new(0.0, 0.1, 0.0),
        velocity: raytracer::vector::Vector3::new(1.0, 2.0, 0.0).normalize() * 11.25,
    };

    for _ in 0..100 {
        tick(&world, &mut particle);
        if particle.position.x >= 100.0 {
            particle.position.x = 99.0;
        }
        if particle.position.x < 0.0 {
            particle.position.x = 0.0;
        }
        if particle.position.y < 0.0 {
            particle.position.y = 0.0;
        }
        if particle.position.y >= 100.0 {
            particle.position.y = 99.0;
            particle.velocity.y *= -1.0;
        }
        canvas.write_pixel(
            particle.position.x as u32,
            particle.position.y as u32,
            raytracer::vector::Color::new(1.0, 0.5, 0.5),
        );
    }

    let ppm = canvas.to_ppm();
    std::fs::write("simple.ppm", ppm).expect("Unable to write file");
}

fn tick(world: &World, particle: &mut Particle) {
    particle.position += particle.velocity;
    particle.velocity = particle.velocity + world.gravity + world.wind;
}
