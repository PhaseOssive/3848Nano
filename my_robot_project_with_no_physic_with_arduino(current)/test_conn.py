import serial
import time
import sys

# --- 配置区 ---
SERIAL_PORT = '/dev/ttyACM0'  # 常见端口1
# SERIAL_PORT = '/dev/ttyUSB0'  # 常见端口2
BAUD_RATE = 9600

def test_connection():
    print(f"--- 串口通信深度测试方案 ---")
    print(f"正在尝试打开端口: {SERIAL_PORT}...")
    
    try:
        # 1. 尝试建立底层连接
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2) # 给 Arduino 重启预留时间
        
        # 2. 发送测试信号
        test_msg = "HELLO_ARDUINO"
        print(f"已物理连接！正在发送测试信号: '{test_msg}'")
        ser.write(f"{test_msg}\n".encode())
        
        # 3. 等待并读取回复
        response = ser.readline().decode().strip()
        
        if response == f"ECHO_BACK:{test_msg}":
            print("\n✅ 【成功连接！】")
            print(f"   Arduino 已通过双向验证，回复内容正确: {response}")
        elif response:
            print("\n⚠️ 【连接异常】")
            print(f"   收到了回复但内容不匹配。回复为: {response}")
            print(f"   可能原因：波特率不匹配或 Arduino 逻辑错误。")
        else:
            print("\n❌ 【连接失败】")
            print(f"   物理端口已打开，但 Arduino 没有任何回复。")
            print(f"   可能原因：线材仅支持供电不支持数据传输，或者 Arduino 程序未运行。")
            
        ser.close()

    except serial.SerialException as e:
        print("\n❌ 【权限或物理错误】")
        if "Permission denied" in str(e):
            print(f"   原因：当前用户没有访问串口的权限。")
            print(f"   解决方法：运行 'sudo chmod 666 {SERIAL_PORT}'")
        elif "No such file" in str(e):
            print(f"   原因：找不到端口 {SERIAL_PORT}。")
            print(f"   解决方法：插拔 USB 线，运行 'ls /dev/tty*' 确认实际端口号。")
        else:
            print(f"   未知串口错误: {e}")
            
    except Exception as e:
        print(f"\n❌ 【系统错误】: {e}")

if __name__ == "__main__":
    test_connection()
    