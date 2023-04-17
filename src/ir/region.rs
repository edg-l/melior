use std::{cell::RefCell, rc::Rc};

use super::Block;
use crate::{
    mlir_sys::{
        mlirBlockGetNextInRegion, mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy,
        mlirRegionEqual, mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlockAfter,
        mlirRegionInsertOwnedBlockBefore, MlirRegion,
    },
    Error,
};

/// A region.
#[derive(Debug)]
pub struct Region {
    pub(crate) raw: MlirRegion,
    pub(crate) owned: bool,
    pub blocks: Vec<Rc<RefCell<Block>>>,
}

impl Region {
    /// Creates a region.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
            owned: true,
            blocks: Vec::default(),
        }
    }

    /// Gets the first block in a region.
    pub fn first_block(&self) -> Option<Rc<RefCell<Block>>> {
        self.blocks.first().cloned()
    }

    /// Gets the last block in a region.
    pub fn last_block(&self) -> Option<Rc<RefCell<Block>>> {
        self.blocks.last().cloned()
    }

    /// Inserts a block after another block.
    pub fn insert_block_after(
        &mut self,
        reference: &Block,
        mut block: Block,
    ) -> Result<Rc<RefCell<Block>>, Error> {
        for (i, b) in self.blocks.iter().enumerate() {
            if *RefCell::borrow(b) == *reference {
                unsafe {
                    mlirRegionInsertOwnedBlockAfter(self.raw, reference.to_raw(), block.to_raw());
                }
                block.owned = false;
                self.blocks.insert(i + 1, Rc::new(RefCell::new(block)));
                return Ok(self.blocks[i].clone());
            }
        }
        Err(Error::BlockNotFound)
    }

    /// Inserts a block before another block.
    pub fn insert_block_before(
        &mut self,
        reference: &Block,
        mut block: Block,
    ) -> Result<Rc<RefCell<Block>>, Error> {
        for (i, b) in self.blocks.iter().enumerate() {
            if *RefCell::borrow(b) == *reference {
                unsafe {
                    mlirRegionInsertOwnedBlockBefore(self.raw, reference.to_raw(), block.to_raw());
                }
                block.owned = false;
                self.blocks.insert(i, Rc::new(RefCell::new(block)));
                return Ok(self.blocks[i].clone());
            }
        }
        Err(Error::BlockNotFound)
    }

    /// Appends a block, returning a reference to it.
    pub fn append_block(&mut self, mut block: Block) -> Rc<RefCell<Block>> {
        unsafe { mlirRegionAppendOwnedBlock(self.raw, block.to_raw()) };
        block.owned = false;
        self.blocks.push(Rc::new(RefCell::new(block)));
        self.blocks.last().unwrap().clone()
    }

    /// Gets this region from the raw handle, population all the blocks, recursively.
    pub(crate) unsafe fn from_raw(raw: MlirRegion, owned: bool) -> Self {
        let mut blocks = Vec::default();

        let mut current_block_raw = unsafe { mlirRegionGetFirstBlock(raw) };

        while !current_block_raw.ptr.is_null() {
            let block = Block::from_raw(current_block_raw, false);
            blocks.push(Rc::new(RefCell::new(block)));
            current_block_raw = unsafe { mlirBlockGetNextInRegion(current_block_raw) };
        }

        Self { raw, owned, blocks }
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.owned {
            unsafe { mlirRegionDestroy(self.raw) }
        }
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for Region {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }

    #[test]
    fn append_block() {
        let mut region = Region::new();
        let block = Block::new(&[]);

        region.append_block(block);

        assert!(region.first_block().is_some());
    }

    #[test]
    fn insert_block_after() {
        let mut region = Region::new();

        let block = region.append_block(Block::new(&[]));
        region
            .insert_block_after(&block.borrow(), Block::new(&[]))
            .unwrap();

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn insert_block_before() {
        let mut region = Region::new();

        let block = region.append_block(Block::new(&[]));
        let block = region
            .insert_block_before(&block.borrow(), Block::new(&[]))
            .unwrap();

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn equal() {
        let region = Region::new();

        assert_eq!(region, region);
    }

    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}
