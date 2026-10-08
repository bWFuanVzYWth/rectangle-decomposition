//! 矩形分解的公共类型。

/// 左闭右开的 `u8` 范围，模仿 `core::ops::Range` 命名。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RangeU8 {
    /// 起始坐标（含）。
    pub start: u8,
    /// 结束坐标（不含）。
    pub end: u8,
}

impl RangeU8 {
    /// 创建一个左闭右开的 `u8` 范围。
    #[must_use]
    pub const fn new(start: u8, end: u8) -> Self {
        Self { start, end }
    }
}

impl From<RangeU8> for core::ops::Range<u8> {
    fn from(r: RangeU8) -> Self {
        r.start..r.end
    }
}

/// 分解结果矩形。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Rectangle {
    /// 矩形覆盖的像素值。
    pub value: u16,
    /// x 范围，左闭右开。
    pub x: RangeU8,
    /// y 范围，左闭右开。
    pub y: RangeU8,
}

/// 调用方持有的固定容量矩形 `SoA` 输出，按 64 字节对齐。
///
/// bounds 和 labels 数组的起点均按 64 字节对齐。
/// 每个矩形占 4 字节 bounds 和 2 字节 label；bounds 的低到高字节依次为
/// `x.start, x.end, y.start, y.end`。坐标采用左闭右开范围，终点可为 64。
/// 所有存储在初始化时置零，清空只修改长度，不分配或重新初始化数组。
#[repr(C, align(64))]
pub struct PackedRectangles64 {
    bounds: [u32; 4096],
    labels: [u16; 4096],
    len: usize,
}

impl Default for PackedRectangles64 {
    fn default() -> Self {
        const { Self::new() }
    }
}

impl PackedRectangles64 {
    #[must_use]
    pub const fn new() -> Self {
        const {
            Self {
                bounds: [0; 4096],
                labels: [0; 4096],
                len: 0,
            }
        }
    }

    pub const fn clear(&mut self) {
        self.len = 0;
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn bounds(&self) -> &[u32] {
        crate::slice(&self.bounds, 0..self.len)
    }

    #[must_use]
    pub fn labels(&self) -> &[u16] {
        crate::slice(&self.labels, 0..self.len)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Rectangle> + '_ {
        self.bounds()
            .iter()
            .zip(self.labels())
            .map(|(&bounds, &value)| {
                let [x_start, x_end, y_start, y_end] = bounds.to_le_bytes();
                Rectangle {
                    value,
                    x: RangeU8::new(x_start, x_end),
                    y: RangeU8::new(y_start, y_end),
                }
            })
    }

    pub(crate) fn push(&mut self, rectangle: Rectangle) -> Result<(), crate::SparseQuadError> {
        if self.len == self.bounds.len() {
            return Err(crate::SparseQuadError::CapacityOverflow);
        }
        *crate::get_mut(&mut self.bounds, self.len) = u32::from_le_bytes([
            rectangle.x.start,
            rectangle.x.end,
            rectangle.y.start,
            rectangle.y.end,
        ]);
        *crate::get_mut(&mut self.labels, self.len) = rectangle.value;
        self.len += 1;
        Ok(())
    }
}

/// 方向枚举。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

/// 有效 chord（矩形边片段），用于 chord 提取和匹配阶段。
///
/// 水平 chord 覆盖 `x1..x2`，垂直 chord 覆盖 `y1..y2`。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectiveChord {
    pub(crate) orientation: Orientation,
    pub(crate) x1: u8,
    pub(crate) y1: u8,
    pub(crate) x2: u8,
    pub(crate) y2: u8,
}

pub trait ChordAccess {
    fn chord(&self) -> EffectiveChord;
}

impl ChordAccess for EffectiveChord {
    fn chord(&self) -> EffectiveChord {
        *self
    }
}

/// 单行上的水平连续段，`x_start..x_end`。
#[derive(Clone, Copy, Default)]
pub struct Run {
    pub(crate) value: u16,
    pub(crate) x_start: u8,
    pub(crate) x_end: u8,
}

/// 向下延伸中的活跃矩形，`x_start..x_end`。
#[derive(Clone, Copy, Default)]
pub struct ActiveRect {
    pub(crate) value: u16,
    pub(crate) x_start: u8,
    pub(crate) x_end: u8,
    pub(crate) y_start: u8,
}

impl ActiveRect {
    pub(crate) const fn new(value: u16, x_start: u8, x_end: u8, y_start: u8) -> Self {
        Self {
            value,
            x_start,
            x_end,
            y_start,
        }
    }

    pub(crate) const fn match_key(self) -> u32 {
        (self.value as u32) | ((self.x_start as u32) << 16) | ((self.x_end as u32) << 24)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod correctness {
        use super::*;

        #[test]
        fn match_key_encoding() {
            // 字段编码：bits[0..16)=value, bits[16..24)=x_start, bits[24..32)=x_end
            let rect = ActiveRect {
                value: 0x1234,
                x_start: 0x56,
                x_end: 0x78,
                y_start: 10,
            };
            assert_eq!(rect.match_key(), 0x7856_1234u32);

            // 不同 value 或 x 范围产生不同 key
            let value_one = ActiveRect {
                value: 1,
                x_start: 10,
                x_end: 20,
                y_start: 0,
            };
            let value_two = ActiveRect {
                value: 2,
                x_start: 10,
                x_end: 20,
                y_start: 0,
            };
            assert_ne!(value_one.match_key(), value_two.match_key());

            let wider = ActiveRect {
                value: 1,
                x_start: 10,
                x_end: 30,
                y_start: 0,
            };
            assert_ne!(value_one.match_key(), wider.match_key());

            // y_start 不参与 key
            let later_y = ActiveRect {
                value: 1,
                x_start: 10,
                x_end: 20,
                y_start: 99,
            };
            assert_eq!(value_one.match_key(), later_y.match_key());
        }
    }
}
