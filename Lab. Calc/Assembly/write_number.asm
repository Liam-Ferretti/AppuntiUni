section .data
    newline db 0Ah

section .bss
    buffer resb 10

section .text
global _start

_start:
    mov eax, 1234
    mov edi, buffer + 10
    mov ecx, 10

convert_loop:
    dec edi
    xor edx, edx
    div ecx ; dividi eax per 10, ovvero l'accumulator, scrivi il quoziente in eax, e il resto in ecx
    add dl, '0'
    mov [edi], dl
    test eax, eax
    jnz convert_loop ; se non 0 allora continua il convert_loop

    mov eax, 4
    mov ebx, 1
    mov ecx, edi
    mov edx, buffer + 10
    sub edx, edi
    int 80h

    mov eax, 4
    mov ebx, 1
    mov ecx, newline
    mov edx, 1
    int 80h

    mov eax, 1
    mov ebx, 0
    int 80h
