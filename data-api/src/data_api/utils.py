import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    # OAuth2PasswordBearer,
    HTTPBearer,
    SecurityScopes,
)

from bla_fastapi.config import get_settings


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str, **kwargs):
        """Returns HTTP 403"""

        super().__init__(status.HTTP_403_FORBIDDEN, detail=detail)


class UnauthenticatedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Requires authentication"
        )


class VerifyToken:
    """Does all the token verification using PyJWT"""

    def __init__(self):
        self.config = get_settings()

        # This gets the JWKS from a given URL and does processing so you can
        # use any of the keys available
        jwks_url = self.config.jwks_url()
        self.jwks_client = jwt.PyJWKClient(jwks_url)

    def verify(
        self,
        security_scopes: SecurityScopes,
        token: HTTPAuthorizationCredentials | None = Depends(
            # OAuth2PasswordBearer(tokenUrl="token")
            HTTPBearer(),
        ),
    ):
        if token is None:
            raise UnauthenticatedException

        # This gets the 'kid' from the passed token
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token).key
        except jwt.exceptions.PyJWKClientError as error:
            raise UnauthorizedException(str(error))
        except jwt.exceptions.DecodeError as error:
            raise UnauthorizedException(str(error))

        try:
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self.config.auth0_algorithms,
                # audience=self.config.auth0_api_audience,
                issuer=self.config.auth0_issuer,
                audience=self.config.auth0_api_audience,
            )
        except Exception as error:
            raise UnauthorizedException(str(error))

        return payload
