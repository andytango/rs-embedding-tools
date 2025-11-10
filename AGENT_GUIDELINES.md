# Rust Development Guidelines

## Core Principles

- Use the simplest possible solution
- Prefer procedural programming and functional programming paradigms
- Use well-maintained, mature crates where possible (check crates.io downloads and recent activity)
- If you feel a need to use complex trait hierarchies or excessive polymorphism, ask the user about your design first
- Ensure you include documentation comments for all public items
- Run clippy as frequently as possible to detect any issues and fix them
- Avoid overriding the linter or using language escape hatches such as:
  - `unsafe` blocks without thorough justification
  - `#[allow(clippy::...)]` attributes
  - `unwrap()` in production code
  - Suppressing compiler warnings
- After you have finished your implementation:
  - Run `cargo build` to check for compilation errors
  - Run `cargo test` to execute tests
  - Run `cargo clippy` for linting
  - If you don't know how to test your implementation, ask the user for help
- When handling unknown data structures, use serde for serialization/deserialization with proper validation

## Rust-Specific Guidelines

### Memory Safety and Ownership

- Understand and follow Rust's ownership rules
- Prefer borrowing (`&T`) over cloning when possible
- Use `Arc` and `Rc` judiciously for shared ownership
- Avoid reference cycles (use `Weak` when needed)
- Prefer stack allocation over heap allocation where possible

### Error Handling

- **Never use `unwrap()` or `expect()` in production code** outside of tests
- Use the `?` operator for error propagation
- Create custom error types with `thiserror` or `anyhow`:
  ```rust
  use thiserror::Error;

  #[derive(Error, Debug)]
  pub enum AppError {
      #[error("IO error: {0}")]
      Io(#[from] std::io::Error),
      #[error("Invalid configuration: {msg}")]
      Config { msg: String },
  }
  ```
- Use `Result<T, E>` as return type for fallible operations
- Consider using `Option<T>` for values that might not exist

### Type System

- Leverage Rust's type system for compile-time guarantees
- Use newtype pattern for domain modeling:
  ```rust
  struct UserId(u64);
  struct Email(String);
  ```
- Prefer enums for state machines and variants
- Use phantom types for additional compile-time safety when needed

### Documentation

- Write documentation comments for all public items:
  ```rust
  /// Calculates the factorial of a number.
  ///
  /// # Arguments
  ///
  /// * `n` - A non-negative integer
  ///
  /// # Returns
  ///
  /// The factorial of `n`
  ///
  /// # Examples
  ///
  /// ```
  /// assert_eq!(factorial(5), 120);
  /// ```
  ///
  /// # Panics
  ///
  /// Panics if the result would overflow
  pub fn factorial(n: u32) -> u32 {
      // implementation
  }
  ```
- Include examples in doc comments
- Document panic conditions
- Document safety requirements for `unsafe` functions

### Code Organization

- Keep modules small and focused
- Use `mod.rs` or separate files for module organization
- Follow Rust naming conventions:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for types, traits
  - `SCREAMING_SNAKE_CASE` for constants
- One type/trait per file for complex items
- Group related functionality in modules

### Traits and Generics

- Implement standard traits where appropriate:
  - `Debug`, `Clone`, `PartialEq`, `Eq`
  - `Default` for types with sensible defaults
  - `From`/`Into` for conversions
  - `Display` for user-facing output
- Use trait bounds judiciously
- Prefer concrete types over generics when generality isn't needed
- Use associated types in traits when appropriate

### Performance

- Profile before optimizing
- Use `&str` instead of `String` for function parameters when possible
- Prefer iterators over collecting into vectors
- Use `Vec::with_capacity` when size is known
- Consider using `SmallVec` for small collections
- Use compile-time computations with `const fn` where possible

### Concurrency

- Prefer message passing with channels over shared state
- Use `Arc<Mutex<T>>` or `Arc<RwLock<T>>` for shared state when necessary
- Consider using `tokio` or `async-std` for async programming
- Avoid blocking in async contexts
- Use `Send` and `Sync` bounds appropriately

### Testing

- Write unit tests in the same file using `#[cfg(test)]`:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_function_name() {
          // test implementation
      }
  }
  ```
- Use integration tests in `tests/` directory
- Use property-based testing with `proptest` or `quickcheck` for complex logic
- Test error cases explicitly
- Use `#[should_panic]` for tests that should panic

### Dependencies

- Prefer well-established crates from the ecosystem
- Common essential crates:
  - `serde` for serialization
  - `tokio` or `async-std` for async runtime
  - `reqwest` for HTTP clients
  - `clap` for CLI arguments
  - `thiserror` or `anyhow` for error handling
  - `tracing` for logging
  - `chrono` or `time` for date/time handling
- Keep dependencies minimal
- Audit dependencies for security with `cargo audit`

### Unsafe Code

- Avoid `unsafe` unless absolutely necessary
- When using `unsafe`:
  - Document safety invariants
  - Keep unsafe blocks as small as possible
  - Write extensive tests
  - Consider using existing safe abstractions
  - Mark containing functions as `unsafe` if they have safety requirements

### Common Patterns

- **Builder Pattern**: For complex object construction
  ```rust
  pub struct ConfigBuilder {
      // fields
  }

  impl ConfigBuilder {
      pub fn new() -> Self { /* ... */ }
      pub fn with_timeout(mut self, timeout: Duration) -> Self { /* ... */ }
      pub fn build(self) -> Result<Config, Error> { /* ... */ }
  }
  ```
- **Type State Pattern**: For compile-time state machines
- **RAII**: Leverage `Drop` trait for cleanup
- **Interior Mutability**: Use `Cell`, `RefCell` when needed

### Validation and Parsing

- Use serde with validation:
  ```rust
  use serde::{Deserialize, Serialize};
  use validator::Validate;

  #[derive(Debug, Deserialize, Serialize, Validate)]
  struct User {
      #[validate(length(min = 1, max = 100))]
      name: String,
      #[validate(email)]
      email: String,
      #[validate(range(min = 0, max = 150))]
      age: u8,
  }
  ```
- Parse, don't validate (make invalid states unrepresentable)
- Use strong types to enforce invariants

### Cargo Configuration

- Use workspace for multi-crate projects
- Configure useful lints in `Cargo.toml`:
  ```toml
  [lints.rust]
  unsafe_code = "warn"
  missing_docs = "warn"

  [lints.clippy]
  all = "warn"
  pedantic = "warn"
  ```
- Use `cargo fmt` for consistent formatting
- Configure `rustfmt.toml` for project style

## Checklist Before Committing

- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes
- [ ] `cargo clippy -- -W clippy::all` shows no warnings
- [ ] `cargo fmt --check` passes
- [ ] No `unwrap()` or `expect()` in production code
- [ ] No `unsafe` blocks without justification and documentation
- [ ] Public API items have documentation comments
- [ ] Examples in documentation compile and run
- [ ] Error handling uses `Result` and `?`
- [ ] No compiler warnings
- [ ] Dependencies are justified and minimal