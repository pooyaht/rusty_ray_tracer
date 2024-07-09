use std::ops::{Add, AddAssign, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

pub type Point = Vector3;
pub type Color = Vector3;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Index<usize> for Vector3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl IntoIterator for Vector3 {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        vec![self.x, self.y, self.z].into_iter()
    }
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let m = self.magnitude();
        Self {
            x: self.x / m,
            y: self.y / m,
            z: self.z / m,
        }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn to_ppm(&self) -> String {
        let clipped_vec3: Vec<f32> = self
            .into_iter()
            .map(|x| {
                if x <= 0.0 {
                    0.0
                } else if x >= 1.0 {
                    255.0
                } else {
                    (x * 255.0).round()
                }
            })
            .collect();
        format!(
            "{} {} {}",
            clipped_vec3[0], clipped_vec3[1], clipped_vec3[2]
        )
    }
}

impl Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl std::ops::Mul<f32> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<f32> for Vector3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::Div<f32> for Vector3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<f32> for Vector3 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul for Vector3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector3_create() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn vector3_add() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        let v3 = v1 + v2;
        assert_eq!(v3.x, 5.0);
        assert_eq!(v3.y, 7.0);
        assert_eq!(v3.z, 9.0);
    }

    #[test]
    fn vector3_add_assign() {
        let mut v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        v1 += v2;
        assert_eq!(v1.x, 5.0);
        assert_eq!(v1.y, 7.0);
        assert_eq!(v1.z, 9.0);
    }

    #[test]
    fn vector3_sub() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        let v3 = v1 - v2;
        assert_eq!(v3.x, -3.0);
        assert_eq!(v3.y, -3.0);
        assert_eq!(v3.z, -3.0);
    }

    #[test]
    fn vector3_sub_assign() {
        let mut v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        v1 -= v2;
        assert_eq!(v1.x, -3.0);
        assert_eq!(v1.y, -3.0);
        assert_eq!(v1.z, -3.0);
    }

    #[test]
    fn vector3_scaler_mul() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = v1 * 2.0;
        assert_eq!(v2.x, 2.0);
        assert_eq!(v2.y, 4.0);
        assert_eq!(v2.z, 6.0);
    }

    #[test]
    fn vector3_scaler_div() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = v1 / 2.0;
        assert_eq!(v2.x, 0.5);
        assert_eq!(v2.y, 1.0);
        assert_eq!(v2.z, 1.5);
    }

    #[test]
    fn vector3_mul_assign() {
        let mut v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = v1 * 2.0;
        v1 *= 2.0;
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
        assert_eq!(v1.z, v2.z);
    }

    #[test]
    fn vector3_div_assign() {
        let mut v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = v1 / 2.0;
        v1 /= 2.0;
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
        assert_eq!(v1.z, v2.z);
    }

    #[test]
    fn vector3_neg() {
        let v1 = -Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v1.x, -1.0);
        assert_eq!(v1.y, -2.0);
        assert_eq!(v1.z, -3.0);
    }

    #[test]
    fn vector3_dot() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);
        assert_eq!(v1.dot(&v2), 32.0);
    }

    #[test]
    fn vector3_mul() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v1 * v2, Vector3::new(1.0, 4.0, 9.0));
    }

    #[test]
    fn vector3_magnitude() {
        let test_cases = [
            (Vector3::new(0.0, 0.0, 0.0), 0.0),
            (Vector3::new(1.0, 0.0, 0.0), 1.0),
            (Vector3::new(0.0, 1.0, 0.0), 1.0),
            (Vector3::new(0.0, 0.0, 1.0), 1.0),
            (Vector3::new(1.0, 1.0, 1.0), 3.0_f32.sqrt()),
        ];
        for (v, expected) in test_cases {
            assert_eq!(v.magnitude(), expected);
        }
    }

    #[test]
    fn vector3_normalize() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let expected = Vector3::new(3.0 / 5.0, 4.0 / 5.0, 0.0 / 5.0);
        assert_eq!(v.normalize(), expected);
    }

    #[test]
    fn vector3_cross() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        let expected = Vector3::new(0.0, 0.0, 1.0);
        assert_eq!(v1.cross(&v2), expected);
    }
}
