use crate::{Context, Error};

use super::{Attribute, Identifier};

/// Helper type. A named attribute, needed on all operations that use attributes.
#[derive(Clone, Copy)]
pub struct NamedAttribute<'c> {
    pub identifier: Identifier<'c>,
    pub attribute: Attribute<'c>,
}

impl<'c> NamedAttribute<'c> {
    pub fn new_parsed(context: &'c Context, name: &str, attribute: &str) -> Result<Self, Error> {
        Ok(Self {
            identifier: Identifier::new(context, name),
            attribute: Attribute::parse(context, attribute)
                .ok_or_else(|| Error::NamedAttributeParse(attribute.to_string()))?,
        })
    }

    pub fn new(identifier: Identifier<'c>, attribute: Attribute<'c>) -> Result<Self, Error> {
        Ok(Self {
            identifier,
            attribute,
        })
    }
}
