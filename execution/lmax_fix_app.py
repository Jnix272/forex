import logging

try:
    import quickfix as fix
except ImportError:
    fix = None

logger = logging.getLogger(__name__)


class LMAXFixApp(fix.Application if fix else object):
    def __init__(self, username: str | None = None, password: str | None = None):
        super().__init__()
        self.username = username
        self.password = password
        self.connected = False
        self.session_id = None

    def onCreate(self, sessionID):
        logger.info(f"[LMAX FIX] Session Created: {sessionID}")
        self.session_id = sessionID

    def onLogon(self, sessionID):
        logger.info(f"[LMAX FIX] Session Logon: {sessionID}")
        self.connected = True

    def onLogout(self, sessionID):
        logger.info(f"[LMAX FIX] Session Logout: {sessionID}")
        self.connected = False

    def toAdmin(self, message, sessionID):
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)

        if msgType.getValue() == fix.MsgType_Logon:
            if self.username:
                message.setField(fix.Username(self.username))
            if self.password:
                message.setField(fix.Password(self.password))
            message.setField(fix.ResetSeqNumFlag(True))

    def fromAdmin(self, message, sessionID):
        pass

    def toApp(self, message, sessionID):
        pass

    def fromApp(self, message, sessionID):
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)

        if msgType.getValue() == fix.MsgType_ExecutionReport:
            clOrdID = fix.ClOrdID()
            ordStatus = fix.OrdStatus()

            cl_id_val = ""
            if message.isSetField(clOrdID):
                message.getField(clOrdID)
                cl_id_val = clOrdID.getValue()

            status_val = ""
            if message.isSetField(ordStatus):
                message.getField(ordStatus)
                status_val = ordStatus.getValue()

            logger.info(f"[LMAX FIX] ExecutionReport: ClOrdID={cl_id_val}, Status={status_val}")
