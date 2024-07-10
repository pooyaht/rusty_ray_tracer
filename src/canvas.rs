use crate::vector::Color;

pub struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<Vec<Color>>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![vec![Color::new(0.0, 0.0, 0.0); width as usize]; height as usize],
        }
    }

    pub fn write_pixel(&mut self, x: u32, y: u32, color: Color) {
        self.pixels[y as usize][x as usize] = color;
    }

    pub fn pixel_at(&self, x: u32, y: u32) -> Color {
        self.pixels[y as usize][x as usize]
    }

    pub fn to_ppm(&self) -> String {
        let mut ppm = format!("P3\n{} {}\n255\n", self.width, self.height);
        let mut current_line = String::new();

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.pixel_at(x, y);
                let color_str = pixel.to_ppm();
                let color_parts: Vec<&str> = color_str.split_whitespace().collect();
                for part in color_parts.iter() {
                    if current_line.len() + part.len() + 1 > 70 {
                        ppm.push_str(&current_line);
                        ppm.push('\n');
                        current_line.clear();
                    }

                    if !current_line.is_empty() {
                        current_line.push(' ');
                    }
                    current_line.push_str(part);
                }
            }
            if !current_line.is_empty() {
                ppm.push_str(&current_line);
                ppm.push('\n');
                current_line.clear();
            }
        }
        ppm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_canvas() {
        let canvas = Canvas::new(10, 20);
        assert_eq!(canvas.pixels.len(), 20);
        assert_eq!(canvas.pixels[0].len(), 10);
        for y in 0..20 {
            for x in 0..10 {
                assert_eq!(canvas.pixel_at(x, y), Color::new(0.0, 0.0, 0.0));
            }
        }
    }

    #[test]
    fn test_write_pixel() {
        let mut canvas = Canvas::new(10, 20);
        canvas.write_pixel(2, 3, Color::new(1.0, 0.0, 0.0));
        assert_eq!(canvas.pixel_at(2, 3), Color::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_pixel_at() {
        let mut canvas = Canvas::new(10, 20);
        canvas.write_pixel(2, 3, Color::new(1.0, 0.0, 0.0));
        assert_eq!(canvas.pixel_at(2, 3), Color::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_to_ppm() {
        let mut canvas = Canvas::new(5, 3);
        canvas.write_pixel(0, 0, Color::new(1.5, 0.0, 0.0));
        canvas.write_pixel(2, 1, Color::new(0.0, 0.5, 0.0));
        canvas.write_pixel(4, 2, Color::new(-0.5, 0.0, 1.0));

        let ppm = canvas.to_ppm();

        assert_eq!(
            ppm,
            "P3\n5 3\n255\n255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0 0 0 0 0 255\n"
        );
    }

    #[test]
    fn test_ends_with_newline() {
        let canvas = Canvas::new(0, 0);
        let ppm = canvas.to_ppm();

        assert_eq!(ppm, "P3\n0 0\n255\n");
    }

    #[test]
    fn test_to_ppm_with_big_numbers() {
        let mut canvas = Canvas::new(10, 2);
        for y in 0..2 {
            for x in 0..10 {
                canvas.write_pixel(x, y, Color::new(1.0, 0.8, 0.6));
            }
        }

        let ppm = canvas.to_ppm();
        assert_eq!(
            ppm,
            "P3\n10 2\n255\n255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n153 255 204 153 255 204 153 255 204 153 255 204 153\n255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n153 255 204 153 255 204 153 255 204 153 255 204 153\n"
        );
    }
}
