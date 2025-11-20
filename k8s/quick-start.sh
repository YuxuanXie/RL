#!/bin/bash

# Kubernetes分布式训练快速启动脚本

set -e

echo "=========================================="
echo "Kubernetes 分布式训练 - 快速启动"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示菜单
show_menu() {
    echo "请选择要启动的训练任务："
    echo ""
    echo "1) PyTorch DQN 分布式训练 (2 workers)"
    echo "2) TensorFlow PPO 分布式训练 (2 workers)"
    echo "3) 查看训练状态"
    echo "4) 查看训练日志"
    echo "5) 停止训练任务"
    echo "6) 清理所有资源"
    echo "0) 退出"
    echo ""
}

# PyTorch训练
start_pytorch() {
    echo -e "${GREEN}[INFO]${NC} 启动PyTorch分布式训练..."
    
    # 检查YAML文件是否存在
    if [ ! -f "k8s/pytorch-distributed-job.yaml" ]; then
        echo -e "${RED}[ERROR]${NC} 找不到 pytorch-distributed-job.yaml"
        exit 1
    fi
    
    # 应用配置
    kubectl apply -f k8s/pytorch-distributed-job.yaml
    
    echo -e "${GREEN}[SUCCESS]${NC} PyTorch训练任务已创建"
    echo ""
    echo "查看状态："
    echo "  kubectl get pods -l app=pytorch-training"
    echo ""
    echo "查看日志："
    echo "  kubectl logs -f pytorch-training-0"
    echo "  kubectl logs -f pytorch-training-1"
}

# TensorFlow训练
start_tensorflow() {
    echo -e "${GREEN}[INFO]${NC} 启动TensorFlow分布式训练..."
    
    # 检查YAML文件是否存在
    if [ ! -f "k8s/tensorflow-distributed-job.yaml" ]; then
        echo -e "${RED}[ERROR]${NC} 找不到 tensorflow-distributed-job.yaml"
        exit 1
    fi
    
    # 应用配置
    kubectl apply -f k8s/tensorflow-distributed-job.yaml
    
    echo -e "${GREEN}[SUCCESS]${NC} TensorFlow训练任务已创建"
    echo ""
    echo "查看状态："
    echo "  kubectl get pods -l app=tf-training"
    echo ""
    echo "查看日志："
    echo "  kubectl logs -f tf-training-0"
    echo "  kubectl logs -f tf-training-1"
}

# 查看状态
check_status() {
    echo -e "${GREEN}[INFO]${NC} 检查训练任务状态..."
    echo ""
    
    echo "=== PyTorch训练任务 ==="
    kubectl get pods -l app=pytorch-training 2>/dev/null || echo "未找到PyTorch训练任务"
    echo ""
    
    echo "=== TensorFlow训练任务 ==="
    kubectl get pods -l app=tf-training 2>/dev/null || echo "未找到TensorFlow训练任务"
    echo ""
    
    echo "=== Services ==="
    kubectl get svc | grep -E "pytorch-distributed|tf-distributed" || echo "未找到相关服务"
    echo ""
}

# 查看日志
view_logs() {
    echo "选择要查看日志的任务："
    echo "1) PyTorch Worker 0"
    echo "2) PyTorch Worker 1"
    echo "3) TensorFlow Worker 0"
    echo "4) TensorFlow Worker 1"
    echo "0) 返回主菜单"
    echo ""
    read -p "请输入选项: " log_choice
    
    case $log_choice in
        1)
            echo -e "${GREEN}[INFO]${NC} 显示 PyTorch Worker 0 日志..."
            kubectl logs -f pytorch-training-0
            ;;
        2)
            echo -e "${GREEN}[INFO]${NC} 显示 PyTorch Worker 1 日志..."
            kubectl logs -f pytorch-training-1
            ;;
        3)
            echo -e "${GREEN}[INFO]${NC} 显示 TensorFlow Worker 0 日志..."
            kubectl logs -f tf-training-0
            ;;
        4)
            echo -e "${GREEN}[INFO]${NC} 显示 TensorFlow Worker 1 日志..."
            kubectl logs -f tf-training-1
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} 无效选项"
            ;;
    esac
}

# 停止训练
stop_training() {
    echo "选择要停止的任务："
    echo "1) PyTorch训练"
    echo "2) TensorFlow训练"
    echo "3) 全部停止"
    echo "0) 返回主菜单"
    echo ""
    read -p "请输入选项: " stop_choice
    
    case $stop_choice in
        1)
            echo -e "${YELLOW}[WARNING]${NC} 停止PyTorch训练..."
            kubectl delete -f k8s/pytorch-distributed-job.yaml 2>/dev/null || echo "PyTorch任务不存在"
            ;;
        2)
            echo -e "${YELLOW}[WARNING]${NC} 停止TensorFlow训练..."
            kubectl delete -f k8s/tensorflow-distributed-job.yaml 2>/dev/null || echo "TensorFlow任务不存在"
            ;;
        3)
            echo -e "${YELLOW}[WARNING]${NC} 停止所有训练任务..."
            kubectl delete -f k8s/pytorch-distributed-job.yaml 2>/dev/null || true
            kubectl delete -f k8s/tensorflow-distributed-job.yaml 2>/dev/null || true
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} 无效选项"
            ;;
    esac
    
    echo -e "${GREEN}[SUCCESS]${NC} 任务已停止"
}

# 清理资源
cleanup() {
    echo -e "${RED}[WARNING]${NC} 这将删除所有训练任务和相关资源！"
    read -p "确认继续？(y/n): " confirm
    
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        return
    fi
    
    echo -e "${YELLOW}[INFO]${NC} 清理资源中..."
    
    # 删除所有资源
    kubectl delete -f k8s/pytorch-distributed-job.yaml 2>/dev/null || true
    kubectl delete -f k8s/tensorflow-distributed-job.yaml 2>/dev/null || true
    
    # 删除孤立的pods
    kubectl delete pods -l app=pytorch-training 2>/dev/null || true
    kubectl delete pods -l app=tf-training 2>/dev/null || true
    
    # 删除services
    kubectl delete svc pytorch-distributed-svc 2>/dev/null || true
    kubectl delete svc tf-distributed-svc 2>/dev/null || true
    
    echo -e "${GREEN}[SUCCESS]${NC} 清理完成"
}

# 主循环
while true; do
    show_menu
    read -p "请输入选项: " choice
    echo ""
    
    case $choice in
        1)
            start_pytorch
            ;;
        2)
            start_tensorflow
            ;;
        3)
            check_status
            ;;
        4)
            view_logs
            ;;
        5)
            stop_training
            ;;
        6)
            cleanup
            ;;
        0)
            echo "退出"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} 无效选项，请重新选择"
            ;;
    esac
    
    echo ""
    read -p "按回车键继续..."
    clear
done
