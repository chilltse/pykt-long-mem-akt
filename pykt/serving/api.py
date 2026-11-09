"""FastAPI adapter. Run one worker; token auth is mandatory unless explicitly in dev mode."""
import asyncio
from contextlib import asynccontextmanager, suppress
import hmac
import logging

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from pykt.models.akt_state_store import StateConflict
from .config import Settings
from .engine import AKTFMKCInferenceEngine
from .errors import ServiceError
from . import schemas as s


def create_app(engine=None, settings=None):
    settings = settings or (engine.settings if engine else Settings.from_env())
    if not settings.dev_auth:
        if len(settings.api_token) < 32 or not settings.allowed_students or not settings.principal:
            raise ValueError('Configure KT_API_TOKEN (>=32 characters), KT_ALLOWED_STUDENTS and KT_PRINCIPAL; '
                             'or explicitly enable KT_DEV_AUTH=true for loopback development')
    if settings.admin_token and (len(settings.admin_token) < 32 or settings.admin_token == settings.api_token):
        raise ValueError('KT_ADMIN_TOKEN must be distinct from KT_API_TOKEN and at least 32 characters')
    allowed = set(settings.allowed_students.split(','))

    @asynccontextmanager
    async def lifespan(app):
        owned = engine is None
        app.state.engine = engine or AKTFMKCInferenceEngine.from_env(settings)
        async def sweep():
            while True:
                await asyncio.sleep(min(30, settings.branch_ttl_seconds))
                def expire():
                    with app.state.engine.lock:
                        app.state.engine.branches.expire()
                await asyncio.to_thread(expire)
        sweeper = asyncio.create_task(sweep())
        try:
            yield
        finally:
            sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await sweeper
            if owned:
                app.state.engine.close()

    app = FastAPI(title='AKT-FMKC Stateful Inference', version='1.0', lifespan=lifespan)

    def get_engine(request: Request):
        return request.app.state.engine

    def auth(authorization: str | None = Header(default=None)):
        if settings.dev_auth:
            return 'development'
        expected = 'Bearer ' + settings.api_token
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise ServiceError('unauthorized', 'Valid bearer token required', 401)
        return settings.principal

    def student_access(student, principal):
        if not settings.dev_auth and student not in allowed and '*' not in allowed:
            raise ServiceError('forbidden', 'Principal cannot access this student', 403)

    def branch_access(e, branch_id, principal):
        info = e.branch_info(branch_id, principal=principal)
        student_access(info['student_id'], principal)

    def names(e, body):
        e.encoder.check_names(body.kc_feature_names)

    def error_response(request, code, message, status, state=None):
        e = getattr(request.app.state, 'engine', None)
        return JSONResponse(status_code=status, content=dict(error=dict(code=code, message=message),
                            model_version=e.model_version if e else None,
                            feature_version=e.feature_version if e else None, state=state))

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error_response(request, exc.code, str(exc), exc.status)

    @app.exception_handler(StateConflict)
    async def conflict(request, exc):
        state = await asyncio.to_thread(request.app.state.engine.get_student_state,
                                        request.path_params['student_id'])
        return error_response(request, 'state_conflict', str(exc), 409, state)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return error_response(request, 'invalid_input', str(exc), 422)

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation(request, exc):
        return error_response(request, 'invalid_request', str(exc), 422)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logging.getLogger(__name__).error('Inference request failed', exc_info=exc)
        return error_response(request, 'internal_error',
                              'Request failed; retry the identical event IDs after checking service logs', 500)

    @app.get('/health')
    def health(e=Depends(get_engine)):
        return dict(status='ok', **e.versions)

    @app.get('/kt/model/info')
    def info(principal=Depends(auth), e=Depends(get_engine)):
        return e.model_info()

    @app.get('/kt/students/{student_id}/state', response_model=s.StateInfo)
    def state(student_id: str, principal=Depends(auth), e=Depends(get_engine)):
        student_access(student_id, principal)
        return e.get_student_state(student_id)

    @app.post('/kt/students/{student_id}/predict', response_model=s.PredictionResult)
    def predict(student_id: str, body: s.PredictRequest, principal=Depends(auth), e=Depends(get_engine)):
        student_access(student_id, principal)
        names(e, body)
        return e.predict_candidates(student_id, body.candidates)

    @app.post('/kt/students/{student_id}/observe', response_model=s.ObservationResult)
    def observe(student_id: str, body: s.ObserveRequest, principal=Depends(auth), e=Depends(get_engine)):
        student_access(student_id, principal)
        names(e, body)
        return e.observe(student_id, body.event, body.expected_revision)

    @app.post('/kt/students/{student_id}/observe-batch', response_model=s.ObservationResult)
    def batch(student_id: str, body: s.BatchRequest, principal=Depends(auth), e=Depends(get_engine)):
        student_access(student_id, principal)
        names(e, body)
        return e.observe_batch(student_id, body.events, body.expected_revision, body.attempt_id)

    @app.post('/kt/branches', response_model=s.BranchInfo)
    def fork(body: s.ForkRequest, principal=Depends(auth), e=Depends(get_engine)):
        student_access(body.student_id, principal)
        return e.fork_student(body.student_id, principal=principal)

    @app.post('/kt/branches/{branch_id}/step', response_model=s.SimulationResult)
    def step(branch_id: str, body: s.StepRequest, principal=Depends(auth), e=Depends(get_engine)):
        branch_access(e, branch_id, principal)
        names(e, body)
        return e.simulate_step(branch_id, body.kc_vector, body.response, principal=principal)

    @app.post('/kt/branches/{branch_id}/predict', response_model=s.BranchPredictionResult)
    def predict_branch(branch_id: str, body: s.PredictRequest, principal=Depends(auth), e=Depends(get_engine)):
        branch_access(e, branch_id, principal)
        names(e, body)
        return e.predict_branch(branch_id, body.candidates, principal=principal)

    @app.delete('/kt/branches/{branch_id}')
    def discard(branch_id: str, principal=Depends(auth), e=Depends(get_engine)):
        branch_access(e, branch_id, principal)
        e.discard_branch(branch_id, principal=principal)
        return dict(deleted=True, **e.versions)

    @app.post('/kt/rollouts/evaluate', response_model=s.RolloutResult)
    def rollout(body: s.RolloutRequest, principal=Depends(auth), e=Depends(get_engine)):
        student_access(body.student_id, principal)
        names(e, body)
        return e.run_rollout(**body.model_dump(exclude={'kc_feature_names'}), principal=principal)

    @app.post('/kt/students/{student_id}/rebuild', response_model=s.StateInfo)
    def rebuild(student_id: str, authorization: str | None = Header(default=None), e=Depends(get_engine)):
        if not settings.admin_token or not hmac.compare_digest(authorization or '', 'Bearer ' + settings.admin_token):
            raise ServiceError('forbidden', 'Separate admin bearer token required', 403)
        # Admin token is intentionally scoped to the same configured student allowlist.
        student_access(student_id, settings.principal)
        return e.rebuild_student(student_id)

    return app
