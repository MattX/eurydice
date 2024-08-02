use std::collections::HashMap;

pub struct Environment<'a, V> {
    parent: Option<&'a Environment<'a, V>>,
    env: HashMap<String, V>,
}

impl<V> Environment<'static, V> {
    pub fn new() -> Self {
        Self {
            parent: None,
            env: HashMap::new(),
        }
    }
}

impl<'a, V> Environment<'a, V> {
    pub fn with_parent(parent: &'a Environment<'a, V>) -> Self {
        Self {
            parent: Some(parent),
            env: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        if let Some(value) = self.env.get(key) {
            Some(value)
        } else if let Some(parent) = self.parent {
            parent.get(key)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: String, value: V) {
        self.env.insert(key, value);
    }
}
