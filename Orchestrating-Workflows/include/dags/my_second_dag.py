from airflow.sdk import dag, task, chain


@dag
def my_second_dag():
    @task
    def my_task_1():
        return 23

    _my_task_1 = my_task_1()

    @task
    def my_task_2():
        return 42

    _my_task_2 = my_task_2()

    @task
    def my_task_3(num1, num2):
        return num1 + num2

    _my_task_3 = my_task_3(num1=_my_task_1, num2=_my_task_2)

    @task
    def my_task_4():
        return "Math!"

    _my_task_4 = my_task_4()

    chain([_my_task_2, _my_task_3], _my_task_4)


my_second_dag()
