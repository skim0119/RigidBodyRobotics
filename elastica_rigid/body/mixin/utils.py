from __future__ import annotations

from typing import Callable, Type


def only_mix_into(
    *required_bases: type,
    allow_direct_subclass: bool = False,
) -> Callable[[Type], Type]:
    """
    Decorate a mixin class to ensure it can only be used in classes that
    are subclasses of one of the provided base classes.

    Enforcement happens when a subclass is created (via __init_subclass__).
    Not yet seen any issues with this approach, but may need to revisit if there are any corner cases with metaclass-based mixins.

    Parameters
    ----------
    *required_bases : type
        One or more base classes that the final composed class must inherit from.
    allow_direct_subclass : bool
        If True, allows direct subclassing of the mixin (i.e., `class X(Mixin): ...`)
        without immediately raising an error. Useful for creating intermediate or abstract
        classes that will later be mixed into a subclass of one of the required bases.
        If False (default), direct subclassing of the mixin is not allowed.
    """
    if len(required_bases) == 0:
        raise ValueError("only_mix_into requires at least one required base class.")

    def decorator(mixin: Type) -> Type:
        prev_init_subclass = getattr(mixin, "__init_subclass__", None)

        # Define a new __init_subclass__ that wraps whatever the mixin had before
        @classmethod
        def __init_subclass__(cls, **kwargs):
            # Call the previous __init_subclass__ if it exists
            if prev_init_subclass is not None:
                prev_init_subclass(**kwargs)  # type: ignore[misc]
            else:
                super(mixin, cls).__init_subclass__(**kwargs)

            # Optionally allow direct subclassing of the mixin itself
            if allow_direct_subclass and cls.__bases__ == (mixin,):
                return

            if not any(issubclass(cls, base) for base in required_bases):
                allowed = ", ".join(base.__name__ for base in required_bases)
                msg = (
                    f"{mixin.__name__} may only be mixed into subclasses of "
                    f"{allowed}; got {cls.__name__}."
                )
                raise TypeError(msg)

        # Attach it
        setattr(mixin, "__init_subclass__", __init_subclass__)
        return mixin

    return decorator
