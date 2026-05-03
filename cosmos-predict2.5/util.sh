until UV_HTTP_TIMEOUT=120 uv sync --extra=cu128; do
    echo "失败了，5秒后重试..."
    sleep 5
done
echo "成功！"