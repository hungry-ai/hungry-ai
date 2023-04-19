from src.backend import Backend


def test_backend(backend: Backend) -> None:
    assert backend.user_service.graph_service == backend.graph_service
    assert backend.image_service.graph_service == backend.graph_service
    assert backend.review_service.graph_service == backend.graph_service

    assert backend.image_service.tag_service == backend.tag_service
