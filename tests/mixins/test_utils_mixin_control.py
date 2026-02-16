import pytest

from elastica_rigid.body.mixin.utils import only_mix_into


@pytest.fixture()
def Base():
    # Define a fresh base class per test to avoid cross-test interference.
    class _Base:
        pass

    return _Base


def test_allows_mixing_into_base_subclass(Base):
    @only_mix_into(Base)
    class Mixin:
        pass

    class Good(Base, Mixin):
        pass

    assert issubclass(Good, Base)
    assert issubclass(Good, Mixin)


def test_rejects_mixing_into_non_base_subclass(Base):
    @only_mix_into(Base)
    class Mixin:
        pass

    with pytest.raises(TypeError):
        class Bad(Mixin):
            pass


def test_allows_direct_subclass_of_mixin_when_enabled(Base):
    @only_mix_into(Base, allow_direct_subclass=True)
    class Mixin:
        pass

    # This should be allowed as an "intermediate" class
    class Intermediate(Mixin):
        pass

    assert issubclass(Intermediate, Mixin)
    assert not issubclass(Intermediate, Base)


def test_forbids_direct_subclass_of_mixin_when_disabled(Base):
    @only_mix_into(Base, allow_direct_subclass=False)
    class StrictMixin:
        pass

    with pytest.raises(TypeError):
        class Intermediate(StrictMixin):
            pass


def test_multiple_inheritance_still_ok_if_base_in_mro(Base):
    class Other:
        pass

    @only_mix_into(Base)
    class Mixin:
        pass

    class Good(Other, Base, Mixin):
        pass

    assert issubclass(Good, Base)
    assert issubclass(Good, Mixin)


def test_allows_mixing_into_any_of_multiple_bases(Base):
    class Base2:
        pass

    @only_mix_into(Base, Base2)
    class Mixin:
        pass

    class Good2(Base2, Mixin):
        pass

    assert issubclass(Good2, Base2)
    assert issubclass(Good2, Mixin)


def test_rejects_when_none_of_multiple_bases_in_mro(Base):
    class Base2:
        pass

    @only_mix_into(Base, Base2)
    class Mixin:
        pass

    with pytest.raises(TypeError):
        class Bad(Mixin):
            pass
