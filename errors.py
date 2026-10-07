"""
地理编码错误模型。

统一全项目的失败信号：查不到地点、算不出几何、模型输出不合法，一律抛
GeocodeError，不再散落裸 ValueError。每个错误带两条信息：

  - user_message：中文、面向使用者，GUI/CLI 直接展示这一条
  - detail：技术细节（英文、含坐标与内部名词），打到控制台供排查

为什么两者要分开存：
  直接把异常原文弹到界面上，用户看到的是 "Unable to find polygon geometry
  for place [西湖] in_region [None]"；而把技术细节整条砍掉，出问题时又
  无从判断是数据源错了还是打分错了。分两个字段，GUI 只显示前者，
  控制台仍能拿到完整的后者。

为什么继承 ValueError：
  GeocodeError 语义上就是"取值无法解析"，且管线里已有 `except ValueError`
  的既有捕获点（parse_text 的重试循环、__main__ 演示入口）。继承它既保持
  语义自洽，也不必在每个捕获点重复列出两个类型。
"""


class GeocodeError(ValueError):
    """地理编码失败：查不到地点、算不出几何，或大模型输出不合法。"""

    def __init__(self, user_message: str, detail: str = "",
                 service_unavailable: bool = False):
        """
        Args:
            user_message: 面向用户的中文提示（一句话说清"哪里出了问题"）。
            detail: 面向开发者的技术细节；省略时退化为 user_message。
            service_unavailable: 这次失败是"数据源没能应答"（超时/被拒），
                而不是"数据源答了：查无此地"。两者对调用方的含义不同：
                查无此地说明已经有权威答复，不能再拿大模型自己编的坐标顶上；
                服务不可用则说明这次压根没问到，只能退回大模型给的粗略范围。
        """
        self.user_message = user_message
        self.detail = detail or user_message
        self.service_unavailable = service_unavailable
        super().__init__(self.detail)
