import logging
from typing import Callable, Any

try:
    import quickfix as fix
except ImportError:
    fix = None

logger = logging.getLogger(__name__)


class LMAXFixApp(fix.Application if fix else object):
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        reset_seq_num: bool = False,
    ):
        super().__init__()
        self.username = username
        self.password = password
        self.reset_seq_num = reset_seq_num
        self.connected = False
        self.session_id = None
        self.executions: dict[str, dict[str, Any]] = {}
        self.on_execution_callback: Callable[[dict[str, Any]], None] | None = None

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
        if not fix:
            return
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)

        if msgType.getValue() == fix.MsgType_Logon:
            if self.username:
                message.setField(fix.Username(self.username))
            if self.password:
                message.setField(fix.Password(self.password))
            if self.reset_seq_num:
                message.setField(fix.ResetSeqNumFlag(True))

    def fromAdmin(self, message, sessionID):
        if not fix:
            return
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        # Log rejects / heartbeats
        if msgType.getValue() == fix.MsgType_Reject:
            text = fix.Text()
            reason = message.getField(text).getValue() if message.isSetField(text) else "unknown"
            logger.warning(f"[LMAX FIX] Admin Reject received: {reason}")

    def toApp(self, message, sessionID):
        pass

    def fromApp(self, message, sessionID):
        if not fix:
            return
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)

        if msgType.getValue() == fix.MsgType_ExecutionReport:
            record: dict[str, Any] = {}
            for field_cls, key in [
                (fix.ClOrdID, "cl_ord_id"),
                (fix.OrderID, "order_id"),
                (fix.OrdStatus, "status"),
                (fix.ExecType, "exec_type"),
                (fix.CumQty, "cum_qty"),
                (fix.LeavesQty, "leaves_qty"),
                (fix.AvgPx, "avg_px"),
                (fix.LastPx, "last_px"),
                (fix.Text, "text"),
            ]:
                f = field_cls()
                if message.isSetField(f):
                    message.getField(f)
                    record[key] = f.getValue()

            cl_id = record.get("cl_ord_id", "")
            if cl_id:
                self.executions[cl_id] = record

            logger.info(f"[LMAX FIX] ExecutionReport: {record}")
            if self.on_execution_callback is not None:
                try:
                    self.on_execution_callback(record)
                except Exception as exc:
                    logger.error(f"[LMAX FIX] execution callback error: {exc}")
